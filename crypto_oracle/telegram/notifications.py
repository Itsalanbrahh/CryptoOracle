"""Proactive Telegram notification builders."""

from __future__ import annotations

from crypto_oracle.models.signals import MasterRecommendation


_AGENT_EMOJI = {
    "Kronos": "📊",
    "Macro": "🌍",
    "Micro": "🔬",
    "Volume": "📦",
    "OnChain": "⛓",
    "Sentiment": "💬",
    "Technical": "📈",
}


def build_alert_message(rec: MasterRecommendation) -> str:
    action_emoji = {"BUY": "🚨", "SELL": "🚨", "HOLD": "ℹ️"}.get(rec.action, "ℹ️")

    lines = [
        f"{action_emoji} *CryptoOracle Alert*",
        f"Symbol: *{rec.symbol}*",
        f"Signal: *{rec.action}* | Confidence: *{rec.confidence*100:.0f}%*",
        "",
        "Top drivers:",
    ]

    top_signals = sorted(
        [s for s in rec.agent_signals if s.signal != "NEUTRAL"],
        key=lambda s: s.confidence,
        reverse=True,
    )[:3]

    if not top_signals:
        top_signals = sorted(rec.agent_signals, key=lambda s: s.confidence, reverse=True)[:3]

    for sig in top_signals:
        emoji = _AGENT_EMOJI.get(sig.agent_name, "•")
        signal_label = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(
            sig.signal, ""
        )
        lines.append(
            f"• {emoji} *{sig.agent_name}*: {signal_label} {sig.signal} "
            f"({sig.confidence*100:.0f}%) — {sig.summary[:80]}"
        )

    if rec.key_risks:
        lines += ["", f"⚠️ *Risks:* {' | '.join(rec.key_risks[:2])}"]

    if rec.suggested_position_size:
        lines += [f"💼 *Suggested size:* {rec.suggested_position_size}"]

    lines += [
        "",
        "Reply with any question or /portfolio to check your positions.",
    ]

    return "\n".join(lines)


def build_status_message(rec: MasterRecommendation) -> str:
    lines = [rec.to_telegram_message(), "", "Agent breakdown:"]
    for sig in rec.agent_signals:
        emoji = _AGENT_EMOJI.get(sig.agent_name, "•")
        signal_label = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(
            sig.signal, "⚪"
        )
        lines.append(
            f"  {emoji} {sig.agent_name}: {signal_label} {sig.signal} "
            f"({sig.confidence*100:.0f}%)"
        )
    return "\n".join(lines)
