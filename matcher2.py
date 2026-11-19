import logging
import os
import threading
import time
from pathlib import Path

# ============================================================
# CPU CONFIGURATION
# Set environment variables BEFORE importing OpenCV / NumPy.
# ============================================================
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["BLIS_NUM_THREADS"] = "1"
try:
    from threadpoolctl import threadpool_limits
    _threadpool_limiter = threadpool_limits(limits=1)
except ImportError:
    _threadpool_limiter = None

import cv2
import numpy as np
from config import SPRITE_DIR

logger = logging.getLogger("poketwo-recognition.timing")
log = logger

# ============================================================
# CPU CONFIGURATION
# ============================================================
OPENCV_THREADS = 1


def configure_cpu_limits():
    """
    Limit OpenCV internal parallelism without restricting
    process CPU affinity.

    Container/hosting controls the actual CPU quota.
    Multiple recognition jobs can execute concurrently.
    """
    cv2.setNumThreads(OPENCV_THREADS)
    cv2.setUseOptimized(True)

    affinity = "unavailable"
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError as exc:
            affinity = f"unavailable: {exc}"

    logger.info(
        "CPU configuration | allowed_cpus=%s | OpenCV threads=%s | threadpoolctl=%s",
        affinity,
        cv2.getNumThreads(),
        "1" if _threadpool_limiter is not None else "unavailable",
    )


# ============================================================
# CONFIGURATION
# ============================================================
ORB_FEATURES_QUERY = 1000
ORB_FEATURES_REFERENCE = 220

ORB_SCALE_FACTOR = 1.15
ORB_LEVELS = 8
ORB_EDGE_THRESHOLD = 12
ORB_PATCH_SIZE = 31
ORB_FAST_THRESHOLD = 10

# Reference images
REFERENCE_MAX_SIDE = 256
QUERY_MAX_SIDE = 480
ALPHA_THRESHOLD = 16

# Retrieval
RETRIEVAL_CANDIDATES = 60
RETRIEVAL_K = 2

# FLANN LSH
LSH_TABLE_NUMBER = 12
LSH_KEY_SIZE = 20
LSH_MULTI_PROBE_LEVEL = 0

# Detailed matching
RATIO_TEST = 0.85
MAX_HAMMING_DISTANCE = 88

MIN_MUTUAL_MATCHES = 4
MIN_RANSAC_INLIERS = 3

MIN_REFERENCE_SPREAD = 0.015
MIN_QUERY_SPREAD = 0.001

RANSAC_REPROJECTION_THRESHOLD = 5.0
RANSAC_MAX_ITERATIONS = 200
RANSAC_CONFIDENCE = 0.99

# Candidate verification
MAX_DETAILED_CANDIDATES = 15

# Logging
LOG_TIMING = True


# ============================================================
# HELPERS
# ============================================================
def _resize_max_side(image, max_side):
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _decode_image(image_bytes):
    if isinstance(image_bytes, np.ndarray):
        image = image_bytes
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _extract_features(orb, image, mask=None, max_features=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None or not keypoints:
        return None
    if max_features is not None and len(keypoints) > max_features:
        responses = np.asarray([kp.response for kp in keypoints], dtype=np.float32)
        indices = np.argpartition(responses, -max_features)[-max_features:]
        indices.sort()
        keypoints = [keypoints[int(i)] for i in indices]
        descriptors = descriptors[indices]
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    return {
        "keypoints": keypoints,
        "points": points,
        "descriptors": np.ascontiguousarray(descriptors),
        "shape": image.shape[:2],
    }


def _entry_value(entry, *keys):
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if value is not None:
                return value
    else:
        for key in keys:
            value = getattr(entry, key, None)
            if value is not None:
                return value
    return None


def _resolve_reference_path(entry, pokemon_id):
    path = _entry_value(entry, "path", "sprite_path", "image_path")
    if path:
        return Path(path)
    return Path(SPRITE_DIR) / f"{pokemon_id}.png"


def _convex_hull_spread(points, width, height):
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    area = cv2.contourArea(hull)
    total_area = max(1.0, float(width * height))
    return float(area / total_area)


def _bbox_spread(points, width, height):
    if len(points) < 2:
        return 0.0
    points = np.asarray(points, dtype=np.float32)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    box_width = max(0.0, float(max_xy[0] - min_xy[0]))
    box_height = max(0.0, float(max_xy[1] - min_xy[1]))
    return (box_width * box_height) / max(1.0, float(width * height))


# ============================================================
# MATCHER
# ============================================================
class PokemonMatcher:
    def __init__(self, entries, limit_cpu=False):
        if limit_cpu:
            configure_cpu_limits()
        self._lsh_lock = threading.Lock()
        self._query_local = threading.local()
        self._matcher_local = threading.local()

        self.reference_orb = cv2.ORB_create(
            nfeatures=ORB_FEATURES_REFERENCE,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_LEVELS,
            edgeThreshold=ORB_EDGE_THRESHOLD,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=ORB_PATCH_SIZE,
            fastThreshold=ORB_FAST_THRESHOLD,
        )
        self.references = []
        self.reference_variants = []
        self.lsh = None
        self.lsh_metadata = []
        self._build_references(entries)
        self._build_lsh_index()

    def _get_query_orb(self):
        orb = getattr(self._query_local, "orb", None)
        if orb is None:
            orb = cv2.ORB_create(
                nfeatures=ORB_FEATURES_QUERY,
                scaleFactor=ORB_SCALE_FACTOR,
                nlevels=ORB_LEVELS,
                edgeThreshold=ORB_EDGE_THRESHOLD,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=ORB_PATCH_SIZE,
                fastThreshold=ORB_FAST_THRESHOLD,
            )
            self._query_local.orb = orb
        return orb

    def _get_bf_matcher(self):
        matcher = getattr(self._matcher_local, "matcher", None)
        if matcher is None:
            matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=False,
            )
            self._matcher_local.matcher = matcher
        return matcher

    def _load_reference_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None, None
        alpha_mask = None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3]
            alpha_mask = np.where(alpha > ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = _resize_max_side(image, REFERENCE_MAX_SIDE)
        if alpha_mask is not None:
            alpha_mask = cv2.resize(
                alpha_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return image, alpha_mask

    def _build_references(self, entries):
        start = time.perf_counter()
        logger.info("Building ORB references...")
        for entry in entries:
            pokemon_id = _entry_value(entry, "id", "pokemon_id", "dex_id")
            if pokemon_id is None:
                continue
            name = _entry_value(entry, "name", "pokemon_name", "title")
            slug = _entry_value(entry, "slug", "key")
            path = _resolve_reference_path(entry, pokemon_id)
            image, alpha_mask = self._load_reference_image(path)
            if image is None:
                continue
            normal = _extract_features(
                self.reference_orb,
                image,
                mask=alpha_mask,
                max_features=ORB_FEATURES_REFERENCE,
            )
            if normal is None:
                continue
            flipped_image = cv2.flip(image, 1)
            flipped_mask = cv2.flip(alpha_mask, 1) if alpha_mask is not None else None
            flipped = _extract_features(
                self.reference_orb,
                flipped_image,
                mask=flipped_mask,
                max_features=ORB_FEATURES_REFERENCE,
            )
            reference = {
                "id": pokemon_id,
                "name": name or str(pokemon_id),
                "slug": slug or str(pokemon_id),
                "path": str(path),
                "normal": normal,
                "flipped": flipped,
            }
            reference_index = len(self.references)
            self.references.append(reference)
            self._add_reference_variant(reference_index, "normal", normal)
            if flipped is not None:
                self._add_reference_variant(reference_index, "flipped", flipped)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "ORB references ready: %s Pokémon, %s variants, %.1fms",
            len(self.references),
            len(self.reference_variants),
            elapsed,
        )

    def _add_reference_variant(self, reference_index, orientation, features):
        if features is None:
            return
        self.reference_variants.append({
            "reference_index": reference_index,
            "orientation": orientation,
            "features": features,
        })

    def _build_lsh_index(self):
        start = time.perf_counter()
        descriptor_arrays = []
        self.lsh_metadata = []
        for variant_index, variant in enumerate(self.reference_variants):
            descriptors = variant["features"]["descriptors"]
            if descriptors is None or len(descriptors) == 0:
                continue
            descriptor_arrays.append(descriptors)
            for descriptor_index in range(len(descriptors)):
                self.lsh_metadata.append((variant_index, descriptor_index))

        if not descriptor_arrays:
            return

        all_descriptors = np.ascontiguousarray(np.vstack(descriptor_arrays), dtype=np.uint8)
        index_params = {
            "algorithm": 6,
            "table_number": LSH_TABLE_NUMBER,
            "key_size": LSH_KEY_SIZE,
            "multi_probe_level": LSH_MULTI_PROBE_LEVEL,
        }
        search_params = {"checks": 64}

        self.lsh = cv2.FlannBasedMatcher(index_params, search_params)
        self.lsh.add([all_descriptors])
        self.lsh.train()

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "FLANN-LSH index ready: %s descriptors, %.1fms",
            len(all_descriptors),
            elapsed,
        )

    def _prepare_query(self, image_bytes):
        image = _decode_image(image_bytes)
        if image is None:
            return None
        image = _resize_max_side(image, QUERY_MAX_SIDE)
        features = _extract_features(
            self._get_query_orb(),
            image,
            max_features=ORB_FEATURES_QUERY,
        )
        if features is None:
            return None
        return {
            "normal": features,
            "image": image,
        }

    def _retrieve_orb_candidates(self, query_features):
        if self.lsh is None:
            return []
        descriptors = query_features["descriptors"]
        if descriptors is None or len(descriptors) == 0:
            return []

        try:
            with self._lsh_lock:
                matches = self.lsh.knnMatch(descriptors, k=RETRIEVAL_K)
        except cv2.error as exc:
            logger.warning("FLANN-LSH retrieval failed: %s", exc)
            return []

        variant_votes = {}
        variant_distances = {}
        for pair in matches:
            if not pair:
                continue
            match = min(pair, key=lambda item: item.distance)
            if match.distance > MAX_HAMMING_DISTANCE:
                continue
            if match.trainIdx >= len(self.lsh_metadata):
                continue
            variant_index, _ = self.lsh_metadata[match.trainIdx]
            variant_votes[variant_index] = variant_votes.get(variant_index, 0) + 1
            variant_distances.setdefault(variant_index, []).append(float(match.distance))

        pokemon_best = {}
        for variant_index, votes in variant_votes.items():
            variant = self.reference_variants[variant_index]
            reference_index = variant["reference_index"]
            orientation = variant["orientation"]
            distances = variant_distances[variant_index]
            mean_distance = float(np.mean(distances))
            retrieval_score = votes / (1.0 + mean_distance)
            previous = pokemon_best.get(reference_index)
            if previous is None or retrieval_score > previous[0]:
                pokemon_best[reference_index] = (retrieval_score, orientation)

        ranked = [
            (score, reference_index, orientation)
            for reference_index, (score, orientation) in pokemon_best.items()
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)

        return [
            (reference_index, orientation)
            for _, reference_index, orientation in ranked[:RETRIEVAL_CANDIDATES]
        ]

    def _mutual_matches(self, query, reference):
        query_descriptors = query["descriptors"]
        reference_descriptors = reference["descriptors"]

        if (
            query_descriptors is None
            or reference_descriptors is None
            or len(query_descriptors) < 2
            or len(reference_descriptors) < 2
        ):
            return []

        matcher = self._get_bf_matcher()

        try:
            forward = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
            reverse = matcher.knnMatch(reference_descriptors, query_descriptors, k=2)
        except cv2.error:
            return []

        forward_good = {}
        for pair in forward:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance > MAX_HAMMING_DISTANCE:
                continue
            if first.distance >= RATIO_TEST * second.distance:
                continue
            forward_good[first.queryIdx] = first

        reverse_good = {}
        for pair in reverse:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance > MAX_HAMMING_DISTANCE:
                continue
            if first.distance >= RATIO_TEST * second.distance:
                continue
            reverse_good[first.queryIdx] = first

        mutual = []
        for query_index, forward_match in forward_good.items():
            reference_index = forward_match.trainIdx
            reverse_match = reverse_good.get(reference_index)
            if reverse_match is None or reverse_match.trainIdx != query_index:
                continue
            mutual.append(forward_match)

        return mutual

    def _verify_geometry(self, query, reference, matches):
        if len(matches) < MIN_MUTUAL_MATCHES:
            return None

        query_points = query["points"]
        reference_points = reference["points"]

        src = np.asarray([reference_points[m.trainIdx] for m in matches], dtype=np.float32)
        dst = np.asarray([query_points[m.queryIdx] for m in matches], dtype=np.float32)

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
            maxIters=RANSAC_MAX_ITERATIONS,
            confidence=RANSAC_CONFIDENCE,
            refineIters=5,
        )

        if matrix is None or inlier_mask is None:
            return None

        inlier_mask = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < MIN_RANSAC_INLIERS:
            return None

        inlier_src = src[inlier_mask]
        inlier_dst = dst[inlier_mask]

        reference_height, reference_width = reference["shape"]
        query_height, query_width = query["shape"]

        reference_spread = _bbox_spread(inlier_src, reference_width, reference_height)
        query_spread = _bbox_spread(inlier_dst, query_width, query_height)
        reference_hull_spread = _convex_hull_spread(inlier_src, reference_width, reference_height)
        query_hull_spread = _convex_hull_spread(inlier_dst, query_width, query_height)

        if reference_spread < MIN_REFERENCE_SPREAD:
            return None
        if query_spread < MIN_QUERY_SPREAD:
            return None

        scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
        if not np.isfinite(scale) or scale <= 0:
            return None

        projected = cv2.transform(inlier_src.reshape(1, -1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - inlier_dst, axis=1)
        mean_error = float(np.mean(errors))

        return {
            "matrix": matrix,
            "inliers": inlier_count,
            "matches": len(matches),
            "inlier_ratio": inlier_count / max(1, len(matches)),
            "mean_error": mean_error,
            "scale": scale,
            "reference_spread": reference_spread,
            "query_spread": query_spread,
            "reference_hull_spread": reference_hull_spread,
            "query_hull_spread": query_hull_spread,
        }

    def _compare_reference(self, query, reference, preferred_orientation="normal"):
        other_orientation = "flipped" if preferred_orientation == "normal" else "normal"

        for orientation in (preferred_orientation, other_orientation):
            reference_features = reference.get(orientation)
            if reference_features is None:
                continue

            matches = self._mutual_matches(query, reference_features)
            geometry = self._verify_geometry(query, reference_features, matches)
            if geometry is None:
                continue

            mean_hamming = float(np.mean([m.distance for m in matches]))
            inlier_score = min(1.0, geometry["inliers"] / 12.0)
            ratio_score = geometry["inlier_ratio"]
            spread_score = min(1.0, geometry["reference_hull_spread"] / 0.20)
            error_score = max(0.0, 1.0 - geometry["mean_error"] / 8.0)
            descriptor_score = max(0.0, 1.0 - mean_hamming / MAX_HAMMING_DISTANCE)

            confidence = (
                0.35 * inlier_score
                + 0.20 * ratio_score
                + 0.15 * spread_score
                + 0.20 * error_score
                + 0.10 * descriptor_score
            )

            return {
                "confidence": float(confidence),
                "distance": float(1.0 - confidence),
                "inliers": geometry["inliers"],
                "matches": geometry["matches"],
                "inlier_ratio": geometry["inlier_ratio"],
                "mean_error": geometry["mean_error"],
                "mean_hamming": mean_hamming,
                "scale": geometry["scale"],
                "mirrored": orientation == "flipped",
            }

        return None

    def identify(self, image_bytes, limit=1):
        total_start = time.perf_counter()
        query_start = time.perf_counter()

        query = self._prepare_query(image_bytes)
        query_ms = (time.perf_counter() - query_start) * 1000

        if query is None:
            return []

        query_features = query["normal"]
        if len(query_features["descriptors"]) < 3:
            return []

        retrieval_start = time.perf_counter()
        candidates = self._retrieve_orb_candidates(query_features)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if not candidates:
            if LOG_TIMING:
                logger.info(
                    "TIMING matcher2 | query=%.1fms | retrieval=%.1fms | candidates=0",
                    query_ms,
                    retrieval_ms,
                )
            return []

        detailed_start = time.perf_counter()
        candidates = candidates[:MAX_DETAILED_CANDIDATES]
        results = []
        for reference_index, preferred_orientation in candidates:
            reference = self.references[reference_index]
            comparison = self._compare_reference(
                query_features,
                reference,
                preferred_orientation=preferred_orientation,
            )
            if comparison is None:
                continue
            results.append({
                "name": reference["name"],
                "id": reference["id"],
                "slug": reference["slug"],
                "path": reference["path"],
                **comparison,
            })
        results.sort(key=lambda result: result["confidence"], reverse=True)

        detailed_ms = (time.perf_counter() - detailed_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        if LOG_TIMING:
            best = results[0] if results else None
            logger.info(
                "TIMING matcher2 | query=%.1fms | retrieval=%.1fms | detailed=%.1fms "
                "| total=%.1fms | features=%s | candidates=%s | verified=%s | best=%s",
                query_ms,
                retrieval_ms,
                detailed_ms,
                total_ms,
                len(query_features["descriptors"]),
                len(candidates),
                len(results),
                best["name"] if best else "NO_RESULT",
            )
            if best:
                logger.info(
                    "MATCH matcher2 | %s | confidence=%.3f | inliers=%s/%s "
                    "| hamming=%.1f | scale=%.3f | mirrored=%s",
                    best["name"],
                    best["confidence"],
                    best["inliers"],
                    best["matches"],
                    best["mean_hamming"],
                    best["scale"],
                    best["mirrored"],
                )

        return results[:max(1, int(limit))]
