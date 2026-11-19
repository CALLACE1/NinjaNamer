import asyncio
import os
import discord
from discord.ext import commands

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.45"))


class NamingCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._match_lock = asyncio.Lock()

    @staticmethod
    def _extract_image_url(message: discord.Message) -> str | None:
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
        session = getattr(self.bot, "http_session", None)
        if not session:
            return None
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
        except Exception:
            return None

    async def _run_identify(self, image_bytes: bytes):
        matcher = getattr(self.bot, "matcher", None)
        if matcher is None:
            return []
        loop = asyncio.get_running_loop()
        async with self._match_lock:
            return await loop.run_in_executor(None, matcher.identify, image_bytes, 1)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        poketwo_id = getattr(self.bot, "poketwo_bot_id", 716390085896962058)
        if message.author.id != poketwo_id:
            return
        if getattr(self.bot, "matcher", None) is None:
            return

        min_conf = MIN_CONFIDENCE
        admin_cog = self.bot.get_cog("AdminCog")
        if admin_cog and message.guild:
            cfg = await admin_cog.get_guild_settings(message.guild.id)
            if not cfg["enabled"]:
                return
            if cfg["channel_ids"] and message.channel.id not in cfg["channel_ids"]:
                return
            min_conf = cfg["min_confidence"]

        image_url = self._extract_image_url(message)
        if not image_url:
            return

        image_bytes = await self._download_bytes(image_url)
        if not image_bytes:
            return

        results = await self._run_identify(image_bytes)
        if not results:
            return

        best = results[0]
        if best["confidence"] < min_conf:
            return

        pct = round(best["confidence"] * 100, 1)
        mirror = " *(mirrored)*" if best.get("mirrored") else ""
        await message.reply(
            f"**{best['name']}** (`#{best['id']}`) — `{pct}%`{mirror}",
            mention_author=False,
        )
        self.bot.dispatch("poketwo_identified", message, best)

    @commands.command(name="id", aliases=["identify", "name"])
    async def identify_cmd(self, ctx: commands.Context):
        target_msg = (
            ctx.message.reference.resolved
            if ctx.message.reference and isinstance(ctx.message.reference.resolved, discord.Message)
            else ctx.message
        )

        image_url = self._extract_image_url(target_msg)
        if not image_url:
            await ctx.reply(
                "Attach an image, paste an image link, or reply to a spawn image.",
                mention_author=False,
            )
            return

        image_bytes = await self._download_bytes(image_url)
        if not image_bytes:
            await ctx.reply("Could not download that image.", mention_author=False)
            return

        results = await self._run_identify(image_bytes)
        if not results:
            await ctx.reply("No confident Pokémon match found.", mention_author=False)
            return

        best = results[0]
        pct = round(best["confidence"] * 100, 1)
        mirror = " *(mirrored)*" if best.get("mirrored") else ""
        await ctx.reply(
            f"**{best['name']}** (`#{best['id']}`) — `{pct}%`{mirror}",
            mention_author=False,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(NamingCog(bot))

