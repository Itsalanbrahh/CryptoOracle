"""Telegram bot: command handlers + conversational agent routing."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from crypto_oracle.models.db import (
    add_stock_symbol,
    add_to_watchlist,
    close_trade,
    get_all_latest,
    get_latest_recommendation,
    get_open_stock_trades,
    get_open_trades,
    get_recommendation_history,
    get_registered_chats,
    get_stock_trade_history,
    get_stock_trade_stats,
    get_stock_watchlist,
    get_trade_history,
    get_trade_stats,
    get_watchlist,
    log_trade,
    register_chat,
    remove_from_watchlist,
    remove_stock_symbol,
    set_alerts,
)
from crypto_oracle.telegram.conversation import handle_free_text
from crypto_oracle.telegram.notifications import build_status_message
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_app: Optional[Application] = None
_run_cooldowns: dict[str, float] = {}
_RUN_COOLDOWN = 300   # 5 minutes


def _check_whitelist(chat_id: str) -> bool:
    allowed_raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    if not allowed_raw.strip():
        return True
    allowed = {s.strip() for s in allowed_raw.split(",")}
    return chat_id in allowed


async def _guard(update: Update) -> bool:
    chat_id = str(update.effective_chat.id)
    if not _check_whitelist(chat_id):
        await update.message.reply_text("Unauthorised.")
        return False
    return True


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    await register_chat(chat_id)
    await update.message.reply_text(
        "👋 *Welcome to CryptoOracle*\n\n"
        "I'm your autonomous crypto + stock trading intelligence system.\n"
        "Running on Alpaca paper trading.\n\n"
        "*Crypto Oracle*\n"
        "/status — latest recommendation\n"
        "/run — trigger fresh oracle analysis\n"
        "/history — last 5 recommendations\n\n"
        "*Crypto Trades*\n"
        "/buy BTC 100 — market buy $100 of BTC\n"
        "/sell BTC — close BTC position\n"
        "/pnl — crypto trade P&L\n\n"
        "*Stock Trading (long/short)*\n"
        "/stocks — stock watchlist + signals\n"
        "/long NVDA 200 — go LONG $200 NVDA\n"
        "/short TSLA 150 — go SHORT $150 TSLA\n"
        "/cover NVDA — close/cover position\n"
        "/stockpnl — stock trade P&L\n"
        "/addstock AAPL · /removestock AAPL\n\n"
        "*Portfolio*\n"
        "/portfolio — Alpaca account summary\n"
        "/report — send PDF dashboard report\n\n"
        "*Auto-Trade*\n"
        "/autotrade on|off|150 — enable/disable/set size\n\n"
        "*Intelligence*\n"
        "/strategy BTC — agent weights & accuracy\n\n"
        "*Settings*\n"
        "/watchlist · /watch ETH · /unwatch ETH\n"
        "/alerts on|off · /interval 120\n\n"
        "Or just ask me anything!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    args = context.args
    symbols = [args[0].upper()] if args else await get_watchlist()
    for symbol in symbols:
        rec = await get_latest_recommendation(symbol)
        if rec is None:
            await update.message.reply_text(
                f"No data for {symbol} yet. Use /run to trigger analysis.",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await update.message.reply_text(
                build_status_message(rec), parse_mode=ParseMode.MARKDOWN
            )


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    now = time.time()
    remaining = _RUN_COOLDOWN - (now - _run_cooldowns.get(chat_id, 0))
    if remaining > 0:
        await update.message.reply_text(
            f"⏳ Cooldown active. Wait {int(remaining // 60)}m {int(remaining % 60)}s.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    _run_cooldowns[chat_id] = now
    await update.message.reply_text("🔄 Running oracle analysis... ~30 seconds.")

    from crypto_oracle.orchestrator import CryptoOracle
    from crypto_oracle.models.db import save_recommendation
    from crypto_oracle.api.websocket import manager
    from crypto_oracle.autotrader import maybe_auto_trade

    args = context.args
    symbols = [args[0].upper()] if args else await get_watchlist()

    oracle = CryptoOracle()
    for symbol in symbols:
        try:
            rec = await oracle.run(symbol)
            rec_id = await save_recommendation(rec)
            rec.id = rec_id
            await manager.broadcast(
                {"type": "recommendation", "data": rec.model_dump(mode="json")}
            )
            trade = await maybe_auto_trade(rec)
            msg = build_status_message(rec)
            if trade:
                if trade.get("action") == "BUY":
                    msg += f"\n\n🤖 *Auto-traded:* BUY ${trade['amount_usd']:.0f} @ ${trade['entry_price']:,.2f}"
                elif trade.get("action") == "SELL":
                    pnl = trade.get("realized_pnl", 0)
                    sign = "+" if pnl >= 0 else ""
                    msg += f"\n\n🤖 *Auto-traded:* SELL | P&L {sign}${pnl:.2f}"
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        except Exception as exc:
            logger.error("Telegram /run failed for %s: %s", symbol, exc, exc_info=True)
            await update.message.reply_text(f"❌ Oracle run failed for {symbol}: {exc}")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    for symbol in await get_watchlist():
        recs = await get_recommendation_history(symbol, limit=5)
        if not recs:
            await update.message.reply_text(f"No history for {symbol}.")
            continue
        lines = [f"📜 *{symbol} — Last {len(recs)} recommendations*\n"]
        for rec in recs:
            ts = rec.timestamp.strftime("%m-%d %H:%M")
            lines.append(
                f"`{ts}` {rec.action} ({rec.confidence*100:.0f}%) — "
                f"{rec.reasoning[:60]}..."
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    try:
        from crypto_oracle.alpaca.client import get_account_summary, get_crypto_positions

        summary = await get_account_summary()
        positions = await get_crypto_positions()
        paper_label = " *(Paper)*" if summary.get("paper_trading") else ""

        lines = [
            f"💰 *Alpaca Account{paper_label}*\n",
            f"Portfolio Value: *${summary.get('portfolio_value', 0):,.2f}*",
            f"Equity: *${summary.get('equity', 0):,.2f}*",
            f"Buying Power: *${summary.get('buying_power', 0):,.2f}*",
            f"Cash: *${summary.get('cash', 0):,.2f}*",
            f"Crypto Value: *${summary.get('crypto_value', 0):,.2f}*",
        ]
        if positions:
            lines += ["", "📊 *Crypto Positions:*"]
            for p in positions:
                sign = "+" if p["unrealized_pl_pct"] >= 0 else ""
                lines.append(
                    f"• {p['symbol']}: {p['quantity']:.6f} "
                    f"@ ${p['current_price']:,.2f} "
                    f"= ${p['market_value']:,.2f} "
                    f"({sign}{p['unrealized_pl_pct']:.1f}%)"
                )
        else:
            lines.append("\n_No open crypto positions._")

        lines.append("\n⚠️ _Paper trading account. Not real money._")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logger.error("Telegram /portfolio failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Portfolio unavailable: {exc}")


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    symbols = await get_watchlist()
    text = "👁 *Watchlist:* " + ", ".join(symbols) if symbols else "Watchlist is empty."
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /watch ETH")
        return
    symbol = context.args[0].upper()
    await add_to_watchlist(symbol)
    await update.message.reply_text(f"✅ Added {symbol} to watchlist.")


async def cmd_unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /unwatch ETH")
        return
    symbol = context.args[0].upper()
    await remove_from_watchlist(symbol)
    await update.message.reply_text(f"✅ Removed {symbol} from watchlist.")


async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    if not context.args or context.args[0].lower() not in ("on", "off"):
        await update.message.reply_text("Usage: /alerts on|off")
        return
    enabled = context.args[0].lower() == "on"
    await set_alerts(chat_id, enabled)
    await update.message.reply_text(
        f"✅ Proactive alerts {'enabled' if enabled else 'disabled'}."
    )


async def cmd_interval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /interval <minutes> (60-1440)")
        return
    try:
        minutes = max(60, min(1440, int(context.args[0])))
    except ValueError:
        await update.message.reply_text("Please provide a number of minutes.")
        return
    from crypto_oracle.scheduler.heartbeat import get_scheduler
    get_scheduler().reschedule_job("oracle_run", trigger="interval", minutes=minutes)
    await update.message.reply_text(f"✅ Oracle interval updated to {minutes} minutes.")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    args = context.args
    symbol = args[0].upper() if args else None
    stats = await get_trade_stats(symbol)
    trades = await get_trade_history(symbol, limit=5)

    sign = "+" if stats["total_pnl"] >= 0 else ""
    lines = [
        f"📊 *Trade Performance{' — ' + symbol if symbol else ''}*\n",
        f"Total P&L: *{sign}${stats['total_pnl']:.2f}*",
        f"Win Rate: *{stats['win_rate']}%* ({stats['winners']}/{stats['closed']} closed)",
        f"Open trades: *{stats['open_count']}*",
    ]
    if stats["closed"]:
        best_sign = "+" if stats["best_trade"] >= 0 else ""
        lines.append(f"Best: *{best_sign}${stats['best_trade']:.2f}* | Worst: *${stats['worst_trade']:.2f}*")

    if trades:
        lines.append("\n*Recent trades:*")
        for t in trades:
            ts = t["created_at"][:10]
            status = "OPEN" if t["status"] == "open" else f"${t['exit_price']:,.0f}"
            pnl_str = ""
            if t["status"] == "closed" and t["realized_pnl"] is not None:
                s = "+" if t["realized_pnl"] >= 0 else ""
                pnl_str = f" → {s}${t['realized_pnl']:.2f}"
            src = "🤖" if t["triggered_by"] == "auto" else "👤"
            lines.append(f"{src} `{ts}` {t['symbol']} @ ${t['entry_price']:,.0f}{pnl_str} [{status}]")
    else:
        lines.append("\n_No trades yet._")

    lines.append("\n⚠️ _Paper trading. Not real money._")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_autotrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    from crypto_oracle.autotrader import get_auto_trade_settings, update_auto_trade_settings

    settings = await get_auto_trade_settings()
    arg = context.args[0].lower() if context.args else "status"

    if arg == "on":
        await update_auto_trade_settings(True, settings["amount_usd"], settings["confidence_threshold"])
        await update.message.reply_text(
            f"✅ Auto-trade *enabled*\n"
            f"Size: *${settings['amount_usd']:.0f}* | Min confidence: *{settings['confidence_threshold']*100:.0f}%*\n"
            f"Oracle will automatically buy/sell on high-confidence signals.",
            parse_mode=ParseMode.MARKDOWN,
        )
    elif arg == "off":
        await update_auto_trade_settings(False, settings["amount_usd"], settings["confidence_threshold"])
        await update.message.reply_text("⏸ Auto-trade *disabled*.", parse_mode=ParseMode.MARKDOWN)
    elif arg.replace(".", "").isdigit():
        amount = float(arg)
        await update_auto_trade_settings(settings["enabled"], amount, settings["confidence_threshold"])
        await update.message.reply_text(
            f"✅ Auto-trade size set to *${amount:.0f}* per trade.", parse_mode=ParseMode.MARKDOWN
        )
    else:
        status = "🟢 ENABLED" if settings["enabled"] else "⏸ DISABLED"
        await update.message.reply_text(
            f"🤖 *Auto-Trade Status: {status}*\n"
            f"Trade size: *${settings['amount_usd']:.0f}* per signal\n"
            f"Min confidence: *{settings['confidence_threshold']*100:.0f}%*\n\n"
            f"Commands:\n"
            f"`/autotrade on` — enable\n"
            f"`/autotrade off` — disable\n"
            f"`/autotrade 150` — set size to $150",
            parse_mode=ParseMode.MARKDOWN,
        )


async def cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: `/buy BTC 100`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Amount must be a number. Usage: `/buy BTC 100`", parse_mode=ParseMode.MARKDOWN)
        return

    await update.message.reply_text(f"⏳ Placing BUY order for ${amount:.0f} of {symbol}…")
    try:
        from crypto_oracle.alpaca.client import get_crypto_price, place_crypto_order
        price = await get_crypto_price(symbol)
        result = await place_crypto_order(symbol, "buy", amount)
        qty = amount / price if price > 0 else 0
        await log_trade(symbol=symbol, amount_usd=amount, entry_price=price,
                        quantity=qty, alpaca_order_id=result["order_id"], triggered_by="manual")
        await update.message.reply_text(
            f"✅ *BUY order submitted*\n"
            f"{symbol}: ${amount:.0f} @ ~${price:,.2f}\n"
            f"Est. qty: {qty:.6f} coins\n"
            f"Order ID: `{result['order_id']}`\n\n"
            f"⚠️ _Paper trading. Not real money._",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.error("Telegram /buy failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Buy failed: {exc}")


async def cmd_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    from crypto_oracle.models.db import get_strategy_state, get_agent_accuracy, get_recommendation_outcomes
    args = context.args
    symbol = args[0].upper() if args else "BTC"

    state   = await get_strategy_state(symbol)
    perf    = await get_agent_accuracy(symbol, limit=20)
    outcomes = await get_recommendation_outcomes(symbol, limit=10)

    correct = sum(1 for o in outcomes if o["outcome"] == "correct")
    total   = len(outcomes)
    hit_str = f"{correct}/{total} ({correct/total*100:.0f}%)" if total else "not enough data yet"

    weights = state["agent_weights"]
    weight_lines = ""
    if weights:
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        weight_lines = "\n".join(
            f"  {'▲' if v > 1.0 else '▼' if v < 1.0 else '─'} {k}: {v:.2f}x"
            for k, v in sorted_w
        )
    else:
        weight_lines = "  All equal (1.0x) — not enough history yet"

    perf_lines = ""
    if perf:
        for agent, s in sorted(perf.items(), key=lambda x: -x[1].get("accuracy_pct", 50)):
            bar = "█" * int(s["accuracy_pct"] / 10)
            perf_lines += f"\n  {agent}: {bar} {s['accuracy_pct']:.0f}% ({s['correct']}/{s['total']})"
    else:
        perf_lines = "\n  No accuracy data yet"

    notes = state["strategy_notes"] or "_No strategy notes yet — run the oracle a few times._"

    await update.message.reply_text(
        f"🧠 *Master Strategy — {symbol}*\n\n"
        f"📊 *Overall accuracy:* {hit_str}\n"
        f"🎯 *Confidence threshold:* {state['confidence_threshold']*100:.0f}%\n"
        f"💵 *Auto-trade size:* ${state['auto_trade_amount']:.0f}\n\n"
        f"*Agent weights (learned):*\n{weight_lines}\n\n"
        f"*Agent accuracy:*{perf_lines}\n\n"
        f"*Strategy notes:*\n_{notes}_",
        parse_mode="Markdown",
    )


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    await update.message.reply_text("📄 Generating report…")
    try:
        from crypto_oracle.telegram.pdf_report import build_report_pdf
        buf = await build_report_pdf()
        now_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
        await update.message.reply_document(
            document=buf,
            filename=f"crypto_oracle_{now_str}.pdf",
            caption="CryptoOracle Dashboard Report",
        )
    except Exception as exc:
        logger.error("Telegram /report failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Report generation failed: {exc}")


async def cmd_sell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    if not context.args:
        await update.message.reply_text("Usage: `/sell BTC`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()

    await update.message.reply_text(f"⏳ Closing {symbol} position…")
    try:
        from crypto_oracle.alpaca.client import close_crypto_position, get_crypto_price
        price = await get_crypto_price(symbol)
        await close_crypto_position(symbol)
        open_trades = await get_open_trades(symbol)
        total_pnl = 0.0
        for t in open_trades:
            qty = t["quantity"] or 0
            entry = t["entry_price"] or price
            pnl = (price - entry) * qty
            total_pnl += pnl
            await close_trade(t["id"], price, round(pnl, 4))
        sign = "+" if total_pnl >= 0 else ""
        await update.message.reply_text(
            f"✅ *SELL order submitted*\n"
            f"{symbol} position closed @ ${price:,.2f}\n"
            f"Realized P&L: *{sign}${total_pnl:.2f}*\n\n"
            f"⚠️ _Paper trading. Not real money._",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.error("Telegram /sell failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Sell failed: {exc}")


async def cmd_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    symbols = await get_stock_watchlist()
    if not symbols:
        await update.message.reply_text("Stock watchlist is empty. Use /addstock NVDA.")
        return

    lines = [f"📈 *Stock Watchlist*\n"]
    for sym in symbols:
        rec = await get_latest_recommendation(sym)
        open_trades = await get_open_stock_trades(sym)
        position_str = ""
        if open_trades:
            t = open_trades[0]
            trade_type = t.get("trade_type", "long").upper()
            position_str = f" | {trade_type} @ ${t.get('entry_price',0):,.2f}"
        if rec:
            action_emoji = "🟢" if rec.action == "BUY" else "🔴" if rec.action == "SELL" else "⚪"
            lines.append(
                f"{action_emoji} *{sym}*: {rec.action} ({rec.confidence*100:.0f}%){position_str}"
            )
        else:
            lines.append(f"⚪ *{sym}*: No data yet{position_str}")

    lines.append("\n/long NVDA 200 · /short TSLA 150 · /cover NVDA")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_long(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually go long on a stock. Usage: /long NVDA 200"""
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: `/long NVDA 200`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Amount must be a number. Usage: `/long NVDA 200`", parse_mode=ParseMode.MARKDOWN)
        return

    await update.message.reply_text(f"⏳ Going LONG ${amount:.0f} {symbol}…")
    try:
        from crypto_oracle.alpaca.client import get_stock_price, place_stock_order, is_market_open
        from crypto_oracle.models.db import log_stock_trade
        if not await is_market_open():
            await update.message.reply_text("⚠️ Market is currently closed. Orders can only be placed during market hours.")
            return
        price = await get_stock_price(symbol)
        result = await place_stock_order(symbol, "buy", amount)
        qty = amount / price if price > 0 else 0
        await log_stock_trade(symbol=symbol, trade_type="long", amount_usd=amount,
                              entry_price=price, quantity=qty,
                              alpaca_order_id=result["order_id"], triggered_by="manual")
        await update.message.reply_text(
            f"✅ *LONG order submitted — {symbol}*\n"
            f"${amount:.0f} @ ~${price:,.2f}\n"
            f"Est. qty: {qty:.4f} shares\n\n"
            f"⚠️ _Paper trading. Not real money._",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.error("Telegram /long failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Long failed: {exc}")


async def cmd_short(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually go short on a stock. Usage: /short TSLA 150"""
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: `/short TSLA 150`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Amount must be a number. Usage: `/short TSLA 150`", parse_mode=ParseMode.MARKDOWN)
        return

    await update.message.reply_text(f"⏳ Going SHORT ${amount:.0f} {symbol}…")
    try:
        from crypto_oracle.alpaca.client import get_stock_price, place_stock_order, is_market_open
        from crypto_oracle.models.db import log_stock_trade
        if not await is_market_open():
            await update.message.reply_text("⚠️ Market is currently closed. Short orders require market hours.")
            return
        price = await get_stock_price(symbol)
        result = await place_stock_order(symbol, "sell", amount)
        qty = amount / price if price > 0 else 0
        await log_stock_trade(symbol=symbol, trade_type="short", amount_usd=amount,
                              entry_price=price, quantity=qty,
                              alpaca_order_id=result["order_id"], triggered_by="manual")
        await update.message.reply_text(
            f"✅ *SHORT order submitted — {symbol}*\n"
            f"${amount:.0f} @ ~${price:,.2f}\n"
            f"Est. qty: {qty:.4f} shares shorted\n\n"
            f"⚠️ _Paper trading. Not real money._",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.error("Telegram /short failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Short failed: {exc}")


async def cmd_cover(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Close/cover a stock position. Usage: /cover NVDA"""
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    if not context.args:
        await update.message.reply_text("Usage: `/cover NVDA`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()

    await update.message.reply_text(f"⏳ Closing {symbol} position…")
    try:
        from crypto_oracle.alpaca.client import close_stock_position, get_stock_price
        from crypto_oracle.models.db import close_stock_trade

        price = await get_stock_price(symbol)
        open_trades = await get_open_stock_trades(symbol)
        if not open_trades:
            await update.message.reply_text(f"No open position in {symbol}.")
            return

        await close_stock_position(symbol)
        total_pnl = 0.0
        trade_type = open_trades[0].get("trade_type", "long")
        for t in open_trades:
            qty = t["quantity"] or 0
            entry = t["entry_price"] or price
            pnl = (price - entry) * qty if t["trade_type"] == "long" else (entry - price) * qty
            total_pnl += pnl
            await close_stock_trade(t["id"], price, round(pnl, 4))

        sign = "+" if total_pnl >= 0 else ""
        label = "LONG closed" if trade_type == "long" else "SHORT covered"
        await update.message.reply_text(
            f"✅ *{label} — {symbol}*\n"
            f"Exit @ ${price:,.2f}\n"
            f"Realized P&L: *{sign}${total_pnl:.2f}*\n\n"
            f"⚠️ _Paper trading. Not real money._",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.error("Telegram /cover failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Cover failed: {exc}")


async def cmd_stockpnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    args = context.args
    symbol = args[0].upper() if args else None
    stats = await get_stock_trade_stats(symbol)
    trades = await get_stock_trade_history(symbol, limit=5)

    sign = "+" if stats["total_pnl"] >= 0 else ""
    lines = [
        f"📊 *Stock Trade P&L{' — ' + symbol if symbol else ''}*\n",
        f"Total P&L: *{sign}${stats['total_pnl']:.2f}*",
        f"Win Rate: *{stats['win_rate']}%* ({stats['winners']}/{stats['closed']} closed)",
        f"Open positions: *{stats['open_count']}*",
    ]

    if trades:
        lines.append("\n*Recent stock trades:*")
        for t in trades:
            ts = t["created_at"][:10]
            trade_type = t.get("trade_type", "long").upper()
            status = "OPEN" if t["status"] == "open" else "CLOSED"
            pnl_str = ""
            if t["status"] == "closed" and t["realized_pnl"] is not None:
                s = "+" if t["realized_pnl"] >= 0 else ""
                pnl_str = f" → {s}${t['realized_pnl']:.2f}"
            src = "🤖" if t["triggered_by"] == "auto" else "👤"
            lines.append(f"{src} `{ts}` {t['symbol']} {trade_type} @ ${t['entry_price']:,.2f}{pnl_str} [{status}]")
    else:
        lines.append("\n_No stock trades yet._")

    lines.append("\n⚠️ _Paper trading. Not real money._")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_addstock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: `/addstock AAPL`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()
    await add_stock_symbol(symbol)
    await update.message.reply_text(f"✅ Added {symbol} to stock watchlist.")


async def cmd_removestock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: `/removestock AAPL`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = context.args[0].upper()
    await remove_stock_symbol(symbol)
    await update.message.reply_text(f"✅ Removed {symbol} from stock watchlist.")


async def cmd_runstock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run stock oracle analysis on demand."""
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    args = context.args
    symbols = [args[0].upper()] if args else await get_stock_watchlist()
    if not symbols:
        await update.message.reply_text("Stock watchlist is empty. Use /addstock NVDA.")
        return

    await update.message.reply_text(f"🔄 Running stock analysis for {', '.join(symbols)}…")
    from crypto_oracle.stock_oracle import StockOracle
    from crypto_oracle.models.db import save_recommendation
    from crypto_oracle.stock_autotrader import maybe_stock_auto_trade

    oracle = StockOracle()
    for symbol in symbols:
        try:
            rec = await oracle.run(symbol)
            rec_id = await save_recommendation(rec)
            rec.id = rec_id
            trade = await maybe_stock_auto_trade(rec)
            action_emoji = "🟢" if rec.action == "BUY" else "🔴" if rec.action == "SELL" else "⚪"
            direction = "LONG" if rec.action == "BUY" else "SHORT" if rec.action == "SELL" else "HOLD"
            msg = (
                f"{action_emoji} *{symbol} Stock Signal*\n"
                f"Signal: *{direction}* ({rec.confidence*100:.0f}%)\n\n"
                f"*Reasoning:* {rec.reasoning}\n\n"
                f"*Catalysts:* {' | '.join(rec.key_catalysts[:3])}\n"
                f"*Risks:* {' | '.join(rec.key_risks[:2])}"
            )
            if trade:
                if trade.get("action") in ("BUY", "SELL"):
                    label = "LONG" if trade.get("trade_type") == "long" else "SHORT"
                    msg += f"\n\n🤖 *Auto-traded:* {label} ${trade['amount_usd']:.0f} @ ${trade['entry_price']:,.2f}"
                elif trade.get("action") == "CLOSE":
                    pnl = trade.get("realized_pnl", 0)
                    s = "+" if pnl >= 0 else ""
                    msg += f"\n\n🤖 *Auto-closed:* P&L {s}${pnl:.2f}"
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        except Exception as exc:
            logger.error("Telegram /runstock failed for %s: %s", symbol, exc, exc_info=True)
            await update.message.reply_text(f"❌ Stock analysis failed for {symbol}: {exc}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    text = (update.message.text or "").strip()
    if not text:
        return
    await update.message.chat.send_action("typing")
    try:
        reply = await handle_free_text(chat_id, text)
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logger.error("handle_message failed: %s", exc, exc_info=True)
        await update.message.reply_text("Something went wrong. Try /status.")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def send_message_to_chat(chat_id: str, text: str) -> None:
    global _app
    if _app is None:
        return
    try:
        await _app.bot.send_message(
            chat_id=int(chat_id),
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.warning("send_message_to_chat(%s) failed: %s", chat_id, exc)


def build_application() -> Application:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("unwatch", cmd_unwatch))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("interval", cmd_interval))
    app.add_handler(CommandHandler("strategy", cmd_strategy))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("trades", cmd_pnl))
    app.add_handler(CommandHandler("autotrade", cmd_autotrade))
    app.add_handler(CommandHandler("buy", cmd_buy))
    app.add_handler(CommandHandler("sell", cmd_sell))
    app.add_handler(CommandHandler("stocks", cmd_stocks))
    app.add_handler(CommandHandler("long", cmd_long))
    app.add_handler(CommandHandler("short", cmd_short))
    app.add_handler(CommandHandler("cover", cmd_cover))
    app.add_handler(CommandHandler("stockpnl", cmd_stockpnl))
    app.add_handler(CommandHandler("addstock", cmd_addstock))
    app.add_handler(CommandHandler("removestock", cmd_removestock))
    app.add_handler(CommandHandler("runstock", cmd_runstock))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


async def start_bot() -> None:
    global _app
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled.")
        return

    # Clear any stale webhook before polling (simple one-shot, non-fatal)
    import aiohttp
    try:
        url = f"https://api.telegram.org/bot{token}/deleteWebhook?drop_pending_updates=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                logger.info("Telegram webhook clear: %s", await resp.text())
    except Exception as exc:
        logger.warning("Webhook pre-clear failed (non-fatal): %s", exc)

    await asyncio.sleep(2)

    for attempt in range(3):
        try:
            _app = build_application()
            await _app.initialize()
            await _app.start()
            await _app.updater.start_polling(drop_pending_updates=True, allowed_updates=["message"])
            logger.info("Telegram bot started (polling)")
            return
        except Exception as exc:
            logger.warning("Telegram start attempt %d/3 failed: %s", attempt + 1, exc)
            await asyncio.sleep(5 * (attempt + 1))

    logger.error("Telegram bot failed to start after 3 attempts — running without it")


async def stop_bot() -> None:
    global _app
    if _app is None:
        return
    await _app.updater.stop()
    await _app.stop()
    await _app.shutdown()
    logger.info("Telegram bot stopped")
