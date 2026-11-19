import asyncio
import os
import re
import sqlite3
from pathlib import Path

import discord
from discord.ext import commands

from config import BASE_DIR
from database import load_reference_database

HUNTS_DB_PATH = os.getenv("HUNTS_DB_PATH", os.path.join(BASE_DIR, "data", "hunts.db"))


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


class HuntsCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.db_path = HUNTS_DB_PATH
        self._db_lock = asyncio.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # Strictly ONE shiny hunt per user per guild.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shiny_hunts (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    pokemon_key TEXT NOT NULL,
                    pokemon_name TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (guild_id, user_id)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shiny_hunts_guild_pokemon
                ON shiny_hunts (guild_id, pokemon_key);
                """
            )

            # MULTIPLE collection entries per user per guild.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_hunts (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    pokemon_key TEXT NOT NULL,
                    pokemon_name TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (guild_id, user_id, pokemon_key)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_collection_hunts_guild_pokemon
                ON collection_hunts (guild_id, pokemon_key);
                """
            )
            conn.commit()

    def _resolve_pokemon_sync(self, raw_name: str) -> tuple[str, str]:
        cleaned = " ".join(raw_name.strip().split())
        target_key = _normalize_key(cleaned)
        for entry in load_reference_database():
            entry_name = str(entry.get("name", ""))
            entry_slug = str(entry.get("slug", ""))
            if target_key in {_normalize_key(entry_name), _normalize_key(entry_slug)}:
                display_name = entry_name or cleaned.title()
                return _normalize_key(display_name), display_name
        return target_key, cleaned.title()

    # --------------------------------------------------------
    # SHINY HUNT DB HELPERS (1 POKEMON PER PERSON)
    # --------------------------------------------------------
    def _set_shiny_hunt_sync(
        self, guild_id: int, user_id: int, raw_name: str
    ) -> tuple[bool, str | None, str]:
        key, display_name = self._resolve_pokemon_sync(raw_name)
        if not key:
            return False, None, ""
        with self._connect() as conn:
            prev_row = conn.execute(
                "SELECT pokemon_key, pokemon_name FROM shiny_hunts WHERE guild_id = ? AND user_id = ?;",
                (guild_id, user_id),
            ).fetchone()
            previous_name = str(prev_row["pokemon_name"]) if prev_row else None

            conn.execute(
                """
                INSERT INTO shiny_hunts (guild_id, user_id, pokemon_key, pokemon_name, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    pokemon_key = excluded.pokemon_key,
                    pokemon_name = excluded.pokemon_name,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (guild_id, user_id, key, display_name),
            )
            conn.commit()
            return True, previous_name, display_name

    def _get_shiny_hunt_sync(self, guild_id: int, user_id: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pokemon_name FROM shiny_hunts WHERE guild_id = ? AND user_id = ?;",
                (guild_id, user_id),
            ).fetchone()
            return str(row["pokemon_name"]) if row else None

    def _get_shiny_hunters_sync(
        self, guild_id: int, pokemon_name: str, pokemon_slug: str
    ) -> list[int]:
        keys = {k for k in (_normalize_key(pokemon_name), _normalize_key(pokemon_slug)) if k}
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        params = [guild_id, *sorted(keys)]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT user_id
                FROM shiny_hunts
                WHERE guild_id = ? AND pokemon_key IN ({placeholders});
                """,
                params,
            ).fetchall()
            return [int(row["user_id"]) for row in rows]

    # --------------------------------------------------------
    # COLLECTION DB HELPERS (MULTIPLE POKEMON PER PERSON)
    # --------------------------------------------------------
    def _add_collection_sync(self, guild_id: int, user_id: int, raw_name: str) -> tuple[bool, str]:
        key, display_name = self._resolve_pokemon_sync(raw_name)
        if not key:
            return False, ""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO collection_hunts (guild_id, user_id, pokemon_key, pokemon_name)
                VALUES (?, ?, ?, ?);
                """,
                (guild_id, user_id, key, display_name),
            )
            conn.commit()
            return cur.rowcount > 0, display_name

    def _remove_collection_sync(self, guild_id: int, user_id: int, raw_name: str) -> tuple[bool, str]:
        key, display_name = self._resolve_pokemon_sync(raw_name)
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM collection_hunts
                WHERE guild_id = ? AND user_id = ? AND pokemon_key = ?;
                """,
                (guild_id, user_id, key),
            )
            conn.commit()
            return cur.rowcount > 0, display_name

    def _list_collection_sync(self, guild_id: int, user_id: int) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pokemon_name
                FROM collection_hunts
                WHERE guild_id = ? AND user_id = ?
                ORDER BY pokemon_name ASC;
                """,
                (guild_id, user_id),
            ).fetchall()
            return [str(row["pokemon_name"]) for row in rows]

    def _clear_collection_sync(self, guild_id: int, user_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM collection_hunts WHERE guild_id = ? AND user_id = ?;",
                (guild_id, user_id),
            )
            conn.commit()
            return cur.rowcount

    def _get_collection_hunters_sync(
        self, guild_id: int, pokemon_name: str, pokemon_slug: str
    ) -> list[int]:
        keys = {k for k in (_normalize_key(pokemon_name), _normalize_key(pokemon_slug)) if k}
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        params = [guild_id, *sorted(keys)]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT user_id
                FROM collection_hunts
                WHERE guild_id = ? AND pokemon_key IN ({placeholders});
                """,
                params,
            ).fetchall()
            return [int(row["user_id"]) for row in rows]

    async def _run_db(self, func, *args):
        loop = asyncio.get_running_loop()
        async with self._db_lock:
            return await loop.run_in_executor(None, func, *args)

    # --------------------------------------------------------
    # !sh COMMAND (SINGLE SHINY HUNT PER PERSON)
    # --------------------------------------------------------
    @commands.command(name="sh", aliases=["shinyhunt", "hunt"])
    @commands.guild_only()
    async def sh_cmd(self, ctx: commands.Context, *, pokemon: str | None = None):
        if pokemon is None or not pokemon.strip():
            current = await self._run_db(self._get_shiny_hunt_sync, ctx.guild.id, ctx.author.id)
            if not current:
                await ctx.reply(
                    "You don't have a shiny hunt set. Use `!sh <pokemon name>` to set one.",
                    mention_author=False,
                )
                return
            await ctx.reply(f"**Your Shiny Hunt:** `{current}`", mention_author=False)
            return

        ok, prev_name, new_name = await self._run_db(
            self._set_shiny_hunt_sync, ctx.guild.id, ctx.author.id, pokemon
        )
        if not ok or not new_name:
            await ctx.reply("Please provide a valid Pokémon name.", mention_author=False)
            return

        if prev_name and prev_name.lower() != new_name.lower():
            await ctx.reply(
                f"Replaced your shiny hunt **`{prev_name}`** with **`{new_name}`**.",
                mention_author=False,
            )
        else:
            await ctx.reply(f"Set your shiny hunt to **`{new_name}`**.", mention_author=False)

    # --------------------------------------------------------
    # !collection / !cl COMMANDS (MULTIPLE PER PERSON)
    # --------------------------------------------------------
    @commands.group(name="collection", aliases=["cl", "col"], invoke_without_command=True)
    @commands.guild_only()
    async def collection_group(self, ctx: commands.Context):
        items = await self._run_db(self._list_collection_sync, ctx.guild.id, ctx.author.id)
        if not items:
            await ctx.reply(
                "Your collection ping list is empty. Use `!cl add <pokemon>` to add Pokémon.",
                mention_author=False,
            )
            return
        formatted = ", ".join(f"`{name}`" for name in items)
        await ctx.reply(f"**Your Collection Pings ({len(items)}):** {formatted}", mention_author=False)

    @collection_group.command(name="add")
    @commands.guild_only()
    async def collection_add(self, ctx: commands.Context, *, pokemon: str):
        added, display_name = await self._run_db(
            self._add_collection_sync, ctx.guild.id, ctx.author.id, pokemon
        )
        if not display_name:
            await ctx.reply("Please provide a valid Pokémon name.", mention_author=False)
            return
        if added:
            await ctx.reply(f"Added **`{display_name}`** to your collection pings.", mention_author=False)
        else:
            await ctx.reply(f"**`{display_name}`** is already in your collection pings.", mention_author=False)

    @collection_group.command(name="remove", aliases=["rm", "del"])
    @commands.guild_only()
    async def collection_remove(self, ctx: commands.Context, *, pokemon: str):
        removed, display_name = await self._run_db(
            self._remove_collection_sync, ctx.guild.id, ctx.author.id, pokemon
        )
        if removed:
            await ctx.reply(f"Removed **`{display_name}`** from your collection pings.", mention_author=False)
        else:
            await ctx.reply(f"**`{display_name}`** wasn't in your collection pings.", mention_author=False)

    @collection_group.command(name="list", aliases=["ls"])
    @commands.guild_only()
    async def collection_list(self, ctx: commands.Context):
        await self.collection_group(ctx)

    @collection_group.command(name="clear")
    @commands.guild_only()
    async def collection_clear(self, ctx: commands.Context):
        count = await self._run_db(self._clear_collection_sync, ctx.guild.id, ctx.author.id)
        await ctx.reply(f"Cleared **{count}** Pokémon from your collection pings.", mention_author=False)

    # --------------------------------------------------------
    # SPAWN PING LISTENER
    # --------------------------------------------------------
    @commands.Cog.listener()
    async def on_poketwo_identified(self, message: discord.Message, best: dict):
        if message.guild is None:
            return

        pokemon_name = str(best.get("name", ""))
        pokemon_slug = str(best.get("slug", ""))

        shiny_ids = await self._run_db(
            self._get_shiny_hunters_sync, message.guild.id, pokemon_name, pokemon_slug
        )
        collection_ids = await self._run_db(
            self._get_collection_hunters_sync, message.guild.id, pokemon_name, pokemon_slug
        )

        # Avoid duplicate pings if someone has the same Pokémon in both lists.
        collection_only_ids = [uid for uid in collection_ids if uid not in set(shiny_ids)]

        lines = []
        if shiny_ids:
            shiny_mentions = " ".join(f"<@{uid}>" for uid in shiny_ids)
            lines.append(f"✨ **Shiny Hunt ({pokemon_name}):** {shiny_mentions}")
        if collection_only_ids:
            col_mentions = " ".join(f"<@{uid}>" for uid in collection_only_ids)
            lines.append(f"📦 **Collection ({pokemon_name}):** {col_mentions}")

        if not lines:
            return

        await message.channel.send(
            "\n".join(lines),
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(HuntsCog(bot))
