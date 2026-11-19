import asyncio
import os
import sqlite3
import time
from pathlib import Path

import discord
from discord.ext import commands

from config import BASE_DIR
from database import build_reference_database, load_reference_database
from matcher2 import PokemonMatcher

SETTINGS_DB_PATH = os.getenv("SETTINGS_DB_PATH", os.path.join(BASE_DIR, "data", "settings.db"))
DEFAULT_MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.45"))


class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.started_at = time.monotonic()
        self.db_path = SETTINGS_DB_PATH
        self._rebuild_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._init_db()

    # --------------------------------------------------------
    # SQLITE WAL SETTINGS STORAGE
    # --------------------------------------------------------
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS guild_settings (
                    guild_id INTEGER PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    pings_enabled INTEGER NOT NULL DEFAULT 1,
                    prefix TEXT,
                    min_confidence REAL NOT NULL DEFAULT 0.45
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS allowed_channels (
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    PRIMARY KEY (guild_id, channel_id)
                );
                """
            )
            conn.commit()

    def _ensure_guild_sync(self, conn: sqlite3.Connection, guild_id: int) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO guild_settings (guild_id, enabled, pings_enabled, prefix, min_confidence)
            VALUES (?, 1, 1, NULL, ?);
            """,
            (guild_id, DEFAULT_MIN_CONFIDENCE),
        )

    def _get_guild_config_sync(self, guild_id: int) -> dict:
        with self._connect() as conn:
            self._ensure_guild_sync(conn, guild_id)
            row = conn.execute(
                "SELECT enabled, pings_enabled, prefix, min_confidence FROM guild_settings WHERE guild_id = ?;",
                (guild_id,),
            ).fetchone()
            channels = conn.execute(
                "SELECT channel_id FROM allowed_channels WHERE guild_id = ? ORDER BY channel_id ASC;",
                (guild_id,),
            ).fetchall()
            conn.commit()
            return {
                "enabled": bool(row["enabled"]) if row else True,
                "pings_enabled": bool(row["pings_enabled"]) if row else True,
                "prefix": row["prefix"] if row else None,
                "min_confidence": float(row["min_confidence"]) if row else DEFAULT_MIN_CONFIDENCE,
                "channel_ids": [int(c["channel_id"]) for c in channels],
            }

    def _set_channel_sync(self, guild_id: int, channel_id: int, replace_all: bool = False) -> bool:
        with self._connect() as conn:
            self._ensure_guild_sync(conn, guild_id)
            if replace_all:
                conn.execute("DELETE FROM allowed_channels WHERE guild_id = ?;", (guild_id,))
            cur = conn.execute(
                "INSERT OR IGNORE INTO allowed_channels (guild_id, channel_id) VALUES (?, ?);",
                (guild_id, channel_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def _remove_channel_sync(self, guild_id: int, channel_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM allowed_channels WHERE guild_id = ? AND channel_id = ?;",
                (guild_id, channel_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def _clear_channels_sync(self, guild_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM allowed_channels WHERE guild_id = ?;", (guild_id,))
            conn.commit()
            return cur.rowcount

    def _set_prefix_sync(self, guild_id: int, prefix: str | None) -> None:
        with self._connect() as conn:
            self._ensure_guild_sync(conn, guild_id)
            conn.execute(
                "UPDATE guild_settings SET prefix = ? WHERE guild_id = ?;",
                (prefix, guild_id),
            )
            conn.commit()

    def _set_confidence_sync(self, guild_id: int, confidence: float) -> None:
        with self._connect() as conn:
            self._ensure_guild_sync(conn, guild_id)
            conn.execute(
                "UPDATE guild_settings SET min_confidence = ? WHERE guild_id = ?;",
                (confidence, guild_id),
            )
            conn.commit()

    def _toggle_field_sync(self, guild_id: int, field: str) -> bool:
        with self._connect() as conn:
            self._ensure_guild_sync(conn, guild_id)
            row = conn.execute(
                f"SELECT {field} FROM guild_settings WHERE guild_id = ?;",
                (guild_id,),
            ).fetchone()
            new_val = 0 if (row and int(row[field]) == 1) else 1
            conn.execute(
                f"UPDATE guild_settings SET {field} = ? WHERE guild_id = ?;",
                (new_val, guild_id),
            )
            conn.commit()
            return bool(new_val)

    async def _run_db(self, func, *args):
        loop = asyncio.get_running_loop()
        async with self._db_lock:
            return await loop.run_in_executor(None, func, *args)

    # Public helper other cogs (naming.py / hunts.py) can call
    async def get_guild_settings(self, guild_id: int) -> dict:
        return await self._run_db(self._get_guild_config_sync, guild_id)

    def get_guild_prefix_sync(self, guild_id: int | None) -> str | None:
        if guild_id is None:
            return None
        return self._get_guild_config_sync(guild_id).get("prefix")

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

    # --------------------------------------------------------
    # SERVER SETUP & CONFIGURATION COMMANDS
    # --------------------------------------------------------
    @commands.command(name="setup", aliases=["settings", "config"])
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def setup_cmd(self, ctx: commands.Context):
        cfg = await self.get_guild_settings(ctx.guild.id)
        default_prefix = os.getenv("BOT_PREFIX", "!")
        active_prefix = cfg["prefix"] or default_prefix
        conf_pct = round(cfg["min_confidence"] * 100, 1)

        if cfg["channel_ids"]:
            ch_display = ", ".join(f"<#{cid}>" for cid in cfg["channel_ids"])
        else:
            ch_display = "All channels (no restriction)"

        embed = discord.Embed(
            title=f"NinjaNamer Setup — {ctx.guild.name}",
            color=discord.Color.green() if cfg["enabled"] else discord.Color.red(),
        )
        embed.add_field(name="Auto-Naming", value="`Enabled`" if cfg["enabled"] else "`Disabled`", inline=True)
        embed.add_field(name="Hunt/Collection Pings", value="`Enabled`" if cfg["pings_enabled"] else "`Disabled`", inline=True)
        embed.add_field(name="Prefix", value=f"`{active_prefix}`", inline=True)
        embed.add_field(name="Min Confidence", value=f"`{conf_pct}%`", inline=True)
        embed.add_field(name="Active Channels", value=ch_display, inline=False)
        embed.set_footer(
            text="Commands: !setchannel, !addchannel, !removechannel, !clearchannels, !setprefix, !setconfidence, !toggle, !togglepings"
        )
        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(name="setchannel")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def setchannel_cmd(self, ctx: commands.Context, channel: discord.TextChannel | None = None):
        target = channel or ctx.channel
        await self._run_db(self._set_channel_sync, ctx.guild.id, target.id, True)
        await ctx.reply(
            f"Auto-naming is now restricted exclusively to {target.mention}. Use `!addchannel #other` to allow more or `!clearchannels` to allow all.",
            mention_author=False,
        )

    @commands.command(name="addchannel")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def addchannel_cmd(self, ctx: commands.Context, channel: discord.TextChannel | None = None):
        target = channel or ctx.channel
        added = await self._run_db(self._set_channel_sync, ctx.guild.id, target.id, False)
        if added:
            await ctx.reply(f"Added {target.mention} to allowed naming channels.", mention_author=False)
        else:
            await ctx.reply(f"{target.mention} is already in the allowed channels list.", mention_author=False)

    @commands.command(name="removechannel", aliases=["rmchannel"])
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def removechannel_cmd(self, ctx: commands.Context, channel: discord.TextChannel | None = None):
        target = channel or ctx.channel
        removed = await self._run_db(self._remove_channel_sync, ctx.guild.id, target.id)
        if removed:
            await ctx.reply(f"Removed {target.mention} from allowed naming channels.", mention_author=False)
        else:
            await ctx.reply(f"{target.mention} wasn't in the allowed channels list.", mention_author=False)

    @commands.command(name="clearchannels")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def clearchannels_cmd(self, ctx: commands.Context):
        count = await self._run_db(self._clear_channels_sync, ctx.guild.id)
        await ctx.reply(
            f"Cleared **{count}** channel restriction(s). NinjaNamer now watches all channels in this server.",
            mention_author=False,
        )

    @commands.command(name="setprefix")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def setprefix_cmd(self, ctx: commands.Context, prefix: str):
        cleaned = prefix.strip()
        if not cleaned or len(cleaned) > 5:
            await ctx.reply("Prefix must be between 1 and 5 characters.", mention_author=False)
            return
        if cleaned.lower() == "reset":
            await self._run_db(self._set_prefix_sync, ctx.guild.id, None)
            await ctx.reply(f"Reset server prefix to default (`{os.getenv('BOT_PREFIX', '!')}`).", mention_author=False)
            return
        await self._run_db(self._set_prefix_sync, ctx.guild.id, cleaned)
        await ctx.reply(f"Updated server prefix to `{cleaned}`.", mention_author=False)

    @commands.command(name="setconfidence", aliases=["setconf"])
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def setconfidence_cmd(self, ctx: commands.Context, percent: float):
        if percent < 10 or percent > 95:
            await ctx.reply("Confidence must be between `10` and `95` (e.g. `!setconfidence 45`).", mention_author=False)
            return
        await self._run_db(self._set_confidence_sync, ctx.guild.id, round(percent / 100.0, 4))
        await ctx.reply(f"Minimum spawn match confidence set to `{percent:.1f}%`.", mention_author=False)

    @commands.command(name="toggle")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def toggle_cmd(self, ctx: commands.Context):
        state = await self._run_db(self._toggle_field_sync, ctx.guild.id, "enabled")
        await ctx.reply(
            f"Auto-naming is now **{'ENABLED' if state else 'DISABLED'}** in this server.",
            mention_author=False,
        )

    @commands.command(name="togglepings")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def togglepings_cmd(self, ctx: commands.Context):
        state = await self._run_db(self._toggle_field_sync, ctx.guild.id, "pings_enabled")
        await ctx.reply(
            f"Shiny hunt & collection pings are now **{'ENABLED' if state else 'DISABLED'}** in this server.",
            mention_author=False,
        )

    # --------------------------------------------------------
    # STATS, RELOAD & DATABASE REBUILD COMMANDS
    # --------------------------------------------------------
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

    @commands.command(name="reloadcog", aliases=["reload"])
    @commands.is_owner()
    async def reloadcog_cmd(self, ctx: commands.Context, cog_name: str):
        ext = cog_name if cog_name.startswith("cogs.") else f"cogs.{cog_name}"
        try:
            await self.bot.reload_extension(ext)
            await ctx.reply(f"Reloaded `{ext}`.", mention_author=False)
        except Exception as exc:
            await ctx.reply(f"Failed to reload `{ext}`: `{exc}`", mention_author=False)

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

    async def cog_command_error(self, ctx: commands.Context, error: commands.CommandError):
        if isinstance(error, commands.MissingPermissions):
            await ctx.reply("You need **Administrator** permission to use this command.", mention_author=False)
        elif isinstance(error, commands.NotOwner):
            await ctx.reply("Only the bot owner can run this command.", mention_author=False)
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.reply(f"Missing argument: `{error.param.name}`.", mention_author=False)
        elif isinstance(error, commands.BadArgument):
            await ctx.reply("Invalid argument provided. Check the channel or number format.", mention_author=False)


async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
