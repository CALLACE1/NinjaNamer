import asyncio
import time
import discord
from discord.ext import commands

from database import build_reference_database, load_reference_database
from matcher2 import PokemonMatcher


class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.started_at = time.monotonic()
        self._rebuild_lock = asyncio.Lock()

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        total = max(0, int(seconds))
        days, rem = divmod(total, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        return f"{minutes}m {secs}s"

    @commands.command(name="stats", aliases=["botstats", "info"])
    async def stats_cmd(self, ctx: commands.Context):
        matcher = getattr(self.bot, "matcher", None)
        entries = load_reference_database()
        uptime = self._format_uptime(time.monotonic() - self.started_at)
        latency_ms = round(self.bot.latency * 1000, 1)

        if matcher is not None:
            ref_count = len(matcher.references)
            variant_count = len(matcher.reference_variants)
            lsh_ready = "Yes" if matcher.lsh is not None else "No"
        else:
            ref_count = 0
            variant_count = 0
            lsh_ready = "No"

        embed = discord.Embed(
            title="NinjaNamer Stats",
            color=discord.Color.blurple(),
        )
        embed.add_field(name="Uptime", value=f"`{uptime}`", inline=True)
        embed.add_field(name="Latency", value=f"`{latency_ms} ms`", inline=True)
        embed.add_field(name="Guilds", value=f"`{len(self.bot.guilds)}`", inline=True)
        embed.add_field(name="Reference Entries", value=f"`{len(entries)}`", inline=True)
        embed.add_field(name="ORB References / Variants", value=f"`{ref_count} / {variant_count}`", inline=True)
        embed.add_field(name="FLANN LSH Ready", value=f"`{lsh_ready}`", inline=True)

        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(name="rebuild", aliases=["refreshdb", "reloaddb"])
    @commands.has_permissions(administrator=True)
    async def rebuild_cmd(self, ctx: commands.Context, force: str = ""):
        if self._rebuild_lock.locked():
            await ctx.reply("A reference database rebuild is already in progress.", mention_author=False)
            return

        force_download = force.lower() in {"force", "--force", "true", "1"}

        async with self._rebuild_lock:
            status_msg = await ctx.reply(
                f"Rebuilding reference database (`force={force_download}`)...",
                mention_author=False,
            )

            try:
                entries = await build_reference_database(force=force_download)
                loop = asyncio.get_running_loop()
                new_matcher = await loop.run_in_executor(None, PokemonMatcher, entries)
                self.bot.matcher = new_matcher

                await status_msg.edit(
                    content=(
                        f"Reference database rebuilt: **{len(entries)}** Pokémon cached "
                        f"and **{len(new_matcher.reference_variants)}** ORB variants indexed."
                    )
                )
            except Exception as exc:
                await status_msg.edit(content=f"Rebuild failed: `{exc}`")

    @rebuild_cmd.error
    async def rebuild_error(self, ctx: commands.Context, error: commands.CommandError):
        if isinstance(error, commands.MissingPermissions):
            await ctx.reply("You need **Administrator** permission to run `!rebuild`.", mention_author=False)


async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
