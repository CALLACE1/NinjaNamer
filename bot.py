import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

from config import POKETWO_BOT_ID
from database import build_reference_database, load_reference_database
from matcher2 import PokemonMatcher

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("NinjaNamer")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DEFAULT_PREFIX = os.getenv("BOT_PREFIX", "!")


def resolve_prefix(bot: commands.Bot, message: discord.Message):
    admin_cog = bot.get_cog("AdminCog")
    if admin_cog is not None and message.guild is not None:
        custom_prefix = admin_cog.get_guild_prefix_sync(message.guild.id)
        if custom_prefix and custom_prefix != DEFAULT_PREFIX:
            return commands.when_mentioned_or(custom_prefix, DEFAULT_PREFIX)(bot, message)
    return commands.when_mentioned_or(DEFAULT_PREFIX)(bot, message)


class NinjaNamerBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True

        super().__init__(
            command_prefix=resolve_prefix,
            intents=intents,
            help_command=None,
        )
        self.poketwo_bot_id = POKETWO_BOT_ID
        self.default_prefix = DEFAULT_PREFIX
        self.matcher: PokemonMatcher | None = None
        self.http_session: aiohttp.ClientSession | None = None

    async def setup_hook(self):
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        )

        entries = load_reference_database()
        if not entries:
            logger.info("No cached reference database found. Building reference database...")
            entries = await build_reference_database()

        loop = asyncio.get_running_loop()
        logger.info("Initializing PokemonMatcher with %s entries...", len(entries))
        self.matcher = await loop.run_in_executor(None, PokemonMatcher, entries)
        logger.info("PokemonMatcher ready.")

        cogs_dir = Path(__file__).resolve().parent / "cogs"
        if cogs_dir.exists():
            for cog_file in sorted(cogs_dir.glob("*.py")):
                if cog_file.stem.startswith("_"):
                    continue
                ext_name = f"cogs.{cog_file.stem}"
                await self.load_extension(ext_name)
                logger.info("Loaded extension: %s", ext_name)

        logger.info("NinjaNamer setup complete.")

    async def on_ready(self):
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)

    async def close(self):
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        await super().close()


def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN is not set. Add it to your .env or hosting environment.")
    bot = NinjaNamerBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
