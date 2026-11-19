import asyncio
import logging
import os

import aiohttp
import discord
from dotenv import load_dotenv

from config import POKETWO_BOT_ID
from database import build_reference_database, load_reference_database
from matcher2 import PokemonMatcher

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("poketwo-namer")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.45"))

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True


class NamingBot(discord.Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matcher: PokemonMatcher | None = None
        self.http_session: aiohttp.ClientSession | None = None
        self._match_lock = asyncio.Lock()

    async def setup_hook(self):
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        )
        entries = load_reference_database()
        if not entries:
            logger.info("No cached reference.json found. Building reference database...")
            entries = await build_reference_database()

        loop = asyncio.get_running_loop()
        logger.info("Initializing PokemonMatcher with %s Pokémon entries...", len(entries))
        self.matcher = await loop.run_in_executor(None, PokemonMatcher, entries)
        logger.info("PokemonMatcher ready.")

    async def close(self):
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        await super().close()

    async def on_ready(self):
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)

    @staticmethod
    def _extract_spawn_image_url(message: discord.Message) -> str | None:
        # Pokétwo wild spawns typically place the Pokémon artwork in an embed image.
        for embed in message.embeds:
            if embed.image and embed.image.url:
                return embed.image.url
            if embed.thumbnail and embed.thumbnail.url:
                return embed.thumbnail.url

        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                return attachment.url

        return None

    async def _download_bytes(self, url: str) -> bytes | None:
        if not self.http_session:
            return None
        try:
            async with self.http_session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return None

    async def on_message(self, message: discord.Message):
        if message.author.id != POKETWO_BOT_ID:
            return
        if self.matcher is None:
            return

        image_url = self._extract_spawn_image_url(message)
        if not image_url:
            return

        image_bytes = await self._download_bytes(image_url)
        if not image_bytes:
            return

        loop = asyncio.get_running_loop()
        async with self._match_lock:
            results = await loop.run_in_executor(
                None, self.matcher.identify, image_bytes, 1
            )

        if not results:
            return

        best = results[0]
        if best["confidence"] < MIN_CONFIDENCE:
            return

        confidence_pct = round(best["confidence"] * 100, 1)
        mirror_tag = " *(mirrored)*" if best.get("mirrored") else ""

        # Customize this reply format however you want your naming bot to speak.
        await message.reply(
            f"**{best['name']}** (`#{best['id']}`) — `{confidence_pct}%`{mirror_tag}",
            mention_author=False,
        )


def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN is not set. Add it to your .env or host environment.")
    bot = NamingBot(intents=intents)
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
