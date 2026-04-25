"""Terminal display helpers for CryptoOracle CLI output."""

from __future__ import annotations

from typing import Any


ACTION_COLORS = {
    "BUY": "\033[92m",   # green
    "SELL": "\033[91m",  # red
    "HOLD": "\033[93m",  # yellow
}
RESET = "\033[0m"
BOLD = "\033[1m"


def colorize_action(action: str) -> str:
    color = ACTION_COLORS.get(action.upper(), "")
    return f"{BOLD}{color}{action}{RESET}"


def format_recommendation(rec: dict[str, Any]) -> str:
    action = rec.get("action", "UNKNOWN")
    confidence = rec.get("confidence", 0.0) * 100
    symbol = rec.get("symbol", "?")
    reasoning = rec.get("reasoning", "")
    lines = [
        f"\n{'='*60}",
        f"  {BOLD}CryptoOracle — {symbol}{RESET}",
        f"  Action    : {colorize_action(action)}",
        f"  Confidence: {confidence:.1f}%",
        f"  Reasoning : {reasoning[:120]}{'...' if len(reasoning) > 120 else ''}",
        f"{'='*60}",
    ]
    signals = rec.get("agent_signals", [])
    if signals:
        lines.append(f"  {BOLD}Agent Signals:{RESET}")
        for s in signals:
            name = s.get("agent_name", "?")
            sig = s.get("signal", "?")
            conf = s.get("confidence", 0.0) * 100
            lines.append(f"    {name:<20} {colorize_action(sig):<30} {conf:.0f}%")
    lines.append("")
    return "\n".join(lines)


def format_portfolio(portfolio: dict[str, Any]) -> str:
    total = portfolio.get("total_equity", 0)
    buying_power = portfolio.get("buying_power", 0)
    crypto_val = portfolio.get("crypto_value", 0)
    lines = [
        f"\n{'='*60}",
        f"  {BOLD}Robinhood Portfolio{RESET}",
        f"  Total Equity  : ${total:,.2f}",
        f"  Buying Power  : ${buying_power:,.2f}",
        f"  Crypto Value  : ${crypto_val:,.2f}",
        f"{'='*60}\n",
    ]
    return "\n".join(lines)
