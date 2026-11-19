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
PREFIX = os.getenv("BOT_PREFIX", "!")


class NinjaNamerBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True

        super().__init__(
            command_prefix=PREFIX,
            intents=intents,
            help_command=None,
        )
        self.poketwo_bot_id = POKETWO_BOT_ID
        self.matcher: PokemonMatcher | None = None
        self.http_session: aiohttp.ClientSession | None = None

    async def setup_hook(self):
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        )

        entries = load_reference_database()
        if not entries:
            logger.info("Building reference database...")
            entries = await build_reference_database()

        loop = asyncio.get_running_loop()
        self.matcher = await loop.run_in_executor(None, PokemonMatcher, entries)

        cogs_dir = Path(__file__).parent / "cogs"
        if cogs_dir.exists():
            for cog_file in sorted(cogs_dir.glob("*.py")):
                if cog_file.stem.startswith("_"):
                    continue
                await self.load_extension(f"cogs.{cog_file.stem}")

        await self.tree.sync()
        logger.info("NinjaNamer setup complete.")

    async def close(self):
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        await super().close()


def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN is missing in .env or environment.")
    bot = NinjaNamerBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
