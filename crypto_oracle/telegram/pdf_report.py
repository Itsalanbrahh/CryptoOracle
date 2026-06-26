"""Generate a PDF dashboard report from current oracle state."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette ───────────────────────────────────────────────────────────────────

_DARK  = colors.HexColor("#0a0a1f")
_NAVY  = colors.HexColor("#13131f")
_GREEN = colors.HexColor("#00c853")
_RED   = colors.HexColor("#ff1744")
_AMBER = colors.HexColor("#ff9100")
_BLUE  = colors.HexColor("#4488ff")
_SLATE = colors.HexColor("#8892a4")
_TEXT  = colors.HexColor("#c8d0dc")
_WHITE = colors.white
_BORDER = colors.HexColor("#1e1e30")

_ACTION_COLOR = {"BUY": _GREEN, "SELL": _RED, "HOLD": _AMBER}
_SIGNAL_COLOR = {"BULLISH": _GREEN, "BEARISH": _RED, "NEUTRAL": _AMBER}
_AGENT_LABEL  = {
    "Kronos": "Kronos", "Macro": "Macro", "Micro": "Micro",
    "Volume": "Volume", "OnChain": "OnChain",
    "Sentiment": "Sentiment", "Technical": "Technical",
}

# ── Styles ────────────────────────────────────────────────────────────────────

_base = getSampleStyleSheet()

def _style(**kw) -> ParagraphStyle:
    return ParagraphStyle("_", **kw)

_S = {
    "title":    _style(fontName="Helvetica-Bold", fontSize=22, textColor=_GREEN,   alignment=TA_CENTER, spaceAfter=2),
    "subtitle": _style(fontName="Helvetica",      fontSize=9,  textColor=_SLATE,   alignment=TA_CENTER, spaceAfter=8),
    "section":  _style(fontName="Helvetica-Bold", fontSize=11, textColor=_WHITE,   spaceBefore=10, spaceAfter=4),
    "action":   _style(fontName="Helvetica-Bold", fontSize=28, alignment=TA_CENTER),
    "conf":     _style(fontName="Helvetica",      fontSize=11, textColor=_SLATE,   alignment=TA_CENTER, spaceAfter=6),
    "body":     _style(fontName="Helvetica",      fontSize=9,  textColor=_TEXT,    leading=14, spaceAfter=4),
    "bullet":   _style(fontName="Helvetica",      fontSize=9,  textColor=_TEXT,    leading=13, leftIndent=10),
    "label":    _style(fontName="Helvetica-Bold", fontSize=8,  textColor=_SLATE,   spaceAfter=2),
    "mono":     _style(fontName="Courier",        fontSize=8,  textColor=_SLATE),
    "small":    _style(fontName="Helvetica",      fontSize=7,  textColor=_SLATE,   alignment=TA_CENTER),
}

def _hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=6, spaceBefore=2)

def _sp(h: float = 4) -> Spacer:
    return Spacer(1, h * mm)

# ── Table styles ──────────────────────────────────────────────────────────────

_AGENT_TABLE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  _NAVY),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  _SLATE),
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_DARK, _NAVY]),
    ("GRID",        (0, 0), (-1, -1), 0.3, _BORDER),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ("ALIGN",       (2, 0), (2, -1),  "CENTER"),
    ("ALIGN",       (3, 0), (3, -1),  "CENTER"),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  _SLATE),
])

_TRADE_TABLE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  _NAVY),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  _SLATE),
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_DARK, _NAVY]),
    ("GRID",        (0, 0), (-1, -1), 0.3, _BORDER),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  _SLATE),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(n, dec: int = 2) -> str:
    if n is None:
        return "—"
    return f"{float(n):,.{dec}f}"

def _pct(n) -> str:
    if n is None:
        return "—"
    return f"{float(n)*100:.0f}%"

def _colored_para(text: str, color, style_key: str = "body") -> Paragraph:
    s = ParagraphStyle("_c", parent=_S[style_key], textColor=color)
    return Paragraph(text, s)

# ── Section builders ──────────────────────────────────────────────────────────

def _rec_section(rec, elems: list) -> None:
    action_color = _ACTION_COLOR.get(rec["action"], _SLATE)

    # Symbol header
    elems.append(_hr())
    elems.append(Paragraph(f"{rec['symbol']} — MASTER RECOMMENDATION", _S["section"]))

    # Action + confidence in a 2-col layout
    conf_pct = int((rec.get("confidence") or 0) * 100)
    bar_filled = "█" * (conf_pct // 5)
    bar_empty  = "░" * (20 - conf_pct // 5)

    action_style = ParagraphStyle("_a", parent=_S["action"], textColor=action_color)
    table_data = [[
        Paragraph(rec["action"], action_style),
        [
            Paragraph(f"{conf_pct}% Confidence", ParagraphStyle("_cf", parent=_S["conf"], textColor=action_color)),
            Paragraph(f'<font color="#00c853">{bar_filled}</font><font color="#1e1e30">{bar_empty}</font>', _S["mono"]),
            Paragraph(f'Updated: {rec.get("timestamp", "")[:16].replace("T", " ")} UTC', _S["small"]),
        ],
    ]]
    t = Table(table_data, colWidths=[40*mm, None])
    t.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",   (0, 0), (0, 0),   "CENTER"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    elems.append(t)
    elems.append(_sp(3))

    # Reasoning
    elems.append(Paragraph("Reasoning", _S["label"]))
    elems.append(Paragraph(rec.get("reasoning", "—"), _S["body"]))

    # Catalysts + risks side by side
    cats = rec.get("key_catalysts") or []
    risks = rec.get("key_risks") or []
    if cats or risks:
        elems.append(_sp(2))
        left_lines  = [Paragraph("Key Catalysts", _S["label"])] + [Paragraph(f"+ {c}", ParagraphStyle("_g", parent=_S["bullet"], textColor=_GREEN)) for c in cats[:4]]
        right_lines = [Paragraph("Key Risks", _S["label"])]     + [Paragraph(f"- {r}", ParagraphStyle("_r", parent=_S["bullet"], textColor=_RED))   for r in risks[:4]]
        cr_table = Table([[left_lines, right_lines]], colWidths=[85*mm, 85*mm])
        cr_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0)]))
        elems.append(cr_table)

    if rec.get("suggested_position_size"):
        elems.append(_sp(2))
        elems.append(Paragraph(f"Suggested size: {rec['suggested_position_size']}", ParagraphStyle("_ps", parent=_S["body"], textColor=_AMBER)))

    # Agent signals table
    sigs = rec.get("agent_signals") or []
    if sigs:
        elems.append(_sp(4))
        elems.append(Paragraph("Agent Signals", _S["label"]))
        rows = [["Agent", "Signal", "Conf", "Summary"]]
        col_w = [32*mm, 22*mm, 14*mm, None]
        for sig in sigs:
            sig_color = _SIGNAL_COLOR.get(sig.get("signal"), _SLATE)
            sig_para = Paragraph(sig.get("signal", ""), ParagraphStyle("_sc", parent=_S["body"], textColor=sig_color, fontName="Helvetica-Bold"))
            rows.append([
                Paragraph(_AGENT_LABEL.get(sig.get("agent_name", ""), sig.get("agent_name", "")), _S["body"]),
                sig_para,
                Paragraph(f"{int((sig.get('confidence') or 0)*100)}%", ParagraphStyle("_cc", parent=_S["body"], alignment=TA_CENTER, textColor=sig_color)),
                Paragraph((sig.get("summary") or "")[:120], _S["body"]),
            ])
        tbl = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(_AGENT_TABLE_STYLE)
        elems.append(tbl)


def _portfolio_section(portfolio: dict, elems: list) -> None:
    elems.append(_hr())
    elems.append(Paragraph("PORTFOLIO", _S["section"]))
    acct = portfolio.get("account", {})
    if acct:
        summary_data = [
            ["Portfolio Value", f"${_fmt(acct.get('portfolio_value'))}",
             "Equity", f"${_fmt(acct.get('equity'))}"],
            ["Cash", f"${_fmt(acct.get('cash'))}",
             "Crypto Value", f"${_fmt(acct.get('crypto_value'))}"],
        ]
        tbl = Table(summary_data, colWidths=[35*mm, 40*mm, 35*mm, 40*mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("TEXTCOLOR", (0,0), (0,-1), _SLATE),
            ("TEXTCOLOR", (2,0), (2,-1), _SLATE),
            ("TEXTCOLOR", (1,0), (1,-1), _TEXT),
            ("TEXTCOLOR", (3,0), (3,-1), _TEXT),
            ("TOPPADDING",(0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
            ("LEFTPADDING",(0,0),(-1,-1), 0),
        ]))
        elems.append(tbl)
        label = " (Paper)" if acct.get("paper_trading") else ""
        elems.append(Paragraph(f"Alpaca{label}", _S["small"]))

    positions = portfolio.get("positions", [])
    if positions:
        elems.append(_sp(3))
        elems.append(Paragraph("Open Positions", _S["label"]))
        rows = [["Symbol", "Qty", "Entry $", "Current $", "Value", "P&L"]]
        for p in positions:
            pct = float(p.get("unrealized_pl_pct") or 0)
            pl_color = _GREEN if pct >= 0 else _RED
            sign = "+" if pct >= 0 else ""
            rows.append([
                Paragraph(p.get("symbol", ""), ParagraphStyle("_sym", parent=_S["body"], fontName="Helvetica-Bold")),
                p.get("quantity", ""),
                f"${_fmt(p.get('average_entry_price'), 2)}",
                f"${_fmt(p.get('current_price'), 2)}",
                f"${_fmt(p.get('market_value'))}",
                Paragraph(f"{sign}{pct:.1f}%", ParagraphStyle("_pl", parent=_S["body"], textColor=pl_color)),
            ])
        tbl = Table(rows, colWidths=[25*mm, 25*mm, 28*mm, 28*mm, 28*mm, None], repeatRows=1)
        tbl.setStyle(_AGENT_TABLE_STYLE)
        elems.append(tbl)


def _trades_section(stats: dict, trades: list, elems: list) -> None:
    elems.append(_hr())
    elems.append(Paragraph("TRADE PERFORMANCE", _S["section"]))

    pnl = stats.get("total_pnl", 0) or 0
    pnl_color = _GREEN if pnl >= 0 else _RED
    sign = "+" if pnl >= 0 else ""

    stat_data = [[
        [Paragraph(f"{sign}${_fmt(pnl)}", ParagraphStyle("_pnl", parent=_S["action"], textColor=pnl_color, fontSize=20)),
         Paragraph("Total P&L", _S["small"])],
        [Paragraph(f"{stats.get('win_rate', 0):.0f}%", ParagraphStyle("_wr", parent=_S["action"], textColor=_BLUE, fontSize=20)),
         Paragraph("Win Rate", _S["small"])],
        [Paragraph(str(stats.get("closed", 0)), ParagraphStyle("_cl", parent=_S["action"], textColor=_TEXT, fontSize=20)),
         Paragraph("Closed Trades", _S["small"])],
        [Paragraph(str(stats.get("open_count", 0)), ParagraphStyle("_op", parent=_S["action"], textColor=_AMBER, fontSize=20)),
         Paragraph("Open Trades", _S["small"])],
    ]]
    tbl = Table(stat_data, colWidths=[42*mm, 42*mm, 42*mm, 42*mm])
    tbl.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("LINEAFTER", (0,0), (2,0), 0.5, _BORDER),
        ("TOPPADDING",(0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    elems.append(tbl)

    if trades:
        elems.append(_sp(4))
        elems.append(Paragraph("Recent Trades", _S["label"]))
        rows = [["Time", "Symbol", "Entry $", "Exit $", "Qty", "P&L", "Source"]]
        for t in trades[:10]:
            is_open = t.get("status") == "open"
            trade_pnl = t.get("realized_pnl")
            if is_open:
                pnl_para = Paragraph("OPEN", ParagraphStyle("_op", parent=_S["body"], textColor=_AMBER))
                exit_txt = "—"
            else:
                p_val = trade_pnl or 0
                p_color = _GREEN if p_val >= 0 else _RED
                p_sign = "+" if p_val >= 0 else ""
                pnl_para = Paragraph(f"{p_sign}${_fmt(p_val)}", ParagraphStyle("_tp", parent=_S["body"], textColor=p_color))
                exit_txt = f"${_fmt(t.get('exit_price'), 0)}"
            src = "AUTO" if t.get("triggered_by") == "auto" else "manual"
            src_color = _BLUE if src == "AUTO" else _SLATE
            rows.append([
                Paragraph((t.get("created_at") or "")[:10], _S["mono"]),
                Paragraph(t.get("symbol", ""), ParagraphStyle("_sy", parent=_S["body"], fontName="Helvetica-Bold")),
                f"${_fmt(t.get('entry_price'), 0)}",
                exit_txt,
                f"{float(t.get('quantity') or 0):.5f}",
                pnl_para,
                Paragraph(src, ParagraphStyle("_src", parent=_S["body"], textColor=src_color)),
            ])
        tbl = Table(rows, colWidths=[22*mm, 18*mm, 23*mm, 23*mm, 22*mm, 22*mm, None], repeatRows=1)
        tbl.setStyle(_TRADE_TABLE_STYLE)
        elems.append(tbl)


# ── Public entry point ────────────────────────────────────────────────────────

async def build_report_pdf() -> BytesIO:
    """Gather all data and render a PDF. Returns a BytesIO ready to send."""
    from crypto_oracle.models.db import (
        get_all_latest,
        get_trade_history,
        get_trade_stats,
    )
    from crypto_oracle.autotrader import get_auto_trade_settings

    recs = await get_all_latest()
    stats = await get_trade_stats()
    trades = await get_trade_history(limit=10)
    auto_settings = await get_auto_trade_settings()

    portfolio: dict[str, Any] = {}
    import os
    if os.getenv("SKIP_ALPACA", "false").lower() != "true":
        try:
            from crypto_oracle.alpaca.client import get_account_summary, get_crypto_positions
            portfolio = {
                "account": await get_account_summary(),
                "positions": await get_crypto_positions(),
            }
        except Exception:
            pass

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )

    elems: list = []

    # Header
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elems.append(Paragraph("CRYPTO ORACLE", _S["title"]))
    elems.append(Paragraph(f"Oracle Report — {now_str}", _S["subtitle"]))

    # Auto-trade status line
    at_status = "ON" if auto_settings.get("enabled") else "OFF"
    at_color = _GREEN if auto_settings.get("enabled") else _SLATE
    elems.append(Paragraph(
        f'Auto-Trade: <font color="{at_color.hexval()}">{at_status}</font>'
        f'  |  Size: ${auto_settings.get("amount_usd", 100):.0f}'
        f'  |  Min Confidence: {auto_settings.get("confidence_threshold", 0.7)*100:.0f}%',
        ParagraphStyle("_at", parent=_S["small"], fontSize=8, textColor=_SLATE),
    ))
    elems.append(_sp(2))

    # One section per symbol
    for rec in recs:
        _rec_section(rec.model_dump(mode="json"), elems)
        elems.append(_sp(4))

    if not recs:
        elems.append(_hr())
        elems.append(Paragraph("No oracle data yet. Run /run to generate recommendations.", _S["body"]))

    # Portfolio
    _portfolio_section(portfolio, elems)
    elems.append(_sp(4))

    # Trades
    _trades_section(stats, trades, elems)
    elems.append(_sp(4))

    # Footer
    elems.append(_hr())
    elems.append(Paragraph(
        "CryptoOracle — Alpaca Paper Trading — Educational use only. Not financial advice.",
        ParagraphStyle("_ft", parent=_S["small"], textColor=_SLATE),
    ))

    def _first_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(_DARK)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    def _later_pages(canvas, doc):
        _first_page(canvas, doc)

    doc.build(elems, onFirstPage=_first_page, onLaterPages=_later_pages)
    buf.seek(0)
    return buf
