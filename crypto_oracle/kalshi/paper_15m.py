"""Virtual paper account for KXBTC15M + Jev.

Tracks a cash balance, open positions, and settles against a spot proxy at
expiry (Kraken 1m close nearest to ``close_time``). Not BRTI — good enough to
exercise the entry path end-to-end without risking capital.

State file: ``~/.hermes/state/kalshi_15m_paper.json``
Trade log:  ``~/.hermes/state/kalshi_15m_paper_trades.jsonl``
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STATE_PATH = Path.home() / ".hermes" / "state" / "kalshi_15m_paper.json"
_TRADES_PATH = Path.home() / ".hermes" / "state" / "kalshi_15m_paper_trades.jsonl"

DEFAULT_BANKROLL = float(os.getenv("KALSHI_15M_PAPER_BANKROLL", "100"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperPosition:
    ticker: str
    side: str                 # yes | no
    count: int
    entry_price: float
    strike: float
    close_time: str
    edge: float
    confidence: float
    jev_p_up: float
    jev_action: str
    jev_model: str
    spot_at_entry: float
    opened_at: str
    cost_usd: float
    status: str = "open"      # open | settled
    settle_spot: float | None = None
    settled_up: bool | None = None
    pnl_usd: float | None = None
    settled_at: str | None = None


@dataclass
class PaperAccount:
    bankroll_usd: float = DEFAULT_BANKROLL
    cash_usd: float = DEFAULT_BANKROLL
    realized_pnl_usd: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    positions: list[PaperPosition] = field(default_factory=list)
    updated_at: str = field(default_factory=_now)

    @property
    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def equity_usd(self) -> float:
        # Mark open positions at entry cost (conservative; no MTM).
        locked = sum(p.cost_usd for p in self.open_positions)
        return round(self.cash_usd + locked, 2)


def load_account() -> PaperAccount:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        acct = PaperAccount()
        save_account(acct)
        return acct
    raw = json.loads(_STATE_PATH.read_text())
    positions = [PaperPosition(**p) for p in raw.get("positions", [])]
    return PaperAccount(
        bankroll_usd=float(raw.get("bankroll_usd", DEFAULT_BANKROLL)),
        cash_usd=float(raw.get("cash_usd", DEFAULT_BANKROLL)),
        realized_pnl_usd=float(raw.get("realized_pnl_usd", 0.0)),
        trades=int(raw.get("trades", 0)),
        wins=int(raw.get("wins", 0)),
        losses=int(raw.get("losses", 0)),
        positions=positions,
        updated_at=str(raw.get("updated_at") or _now()),
    )


def save_account(acct: PaperAccount) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    acct.updated_at = _now()
    payload = {
        "bankroll_usd": acct.bankroll_usd,
        "cash_usd": round(acct.cash_usd, 2),
        "realized_pnl_usd": round(acct.realized_pnl_usd, 2),
        "trades": acct.trades,
        "wins": acct.wins,
        "losses": acct.losses,
        "equity_usd": acct.equity_usd,
        "updated_at": acct.updated_at,
        "positions": [asdict(p) for p in acct.positions],
    }
    _STATE_PATH.write_text(json.dumps(payload, indent=2))


def _append_trade_log(row: dict[str, Any]) -> None:
    _TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _TRADES_PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def has_open_ticker(acct: PaperAccount, ticker: str) -> bool:
    return any(p.ticker == ticker and p.status == "open" for p in acct.positions)


def open_paper_trade(
    acct: PaperAccount,
    *,
    ticker: str,
    side: str,
    count: int,
    entry_price: float,
    strike: float,
    close_time: str | None,
    edge: float,
    confidence: float,
    jev_p_up: float,
    jev_action: str,
    jev_model: str,
    spot_at_entry: float,
) -> PaperPosition | None:
    """Debit cash and open a virtual position. Returns None if underfunded/duplicate."""
    if count <= 0 or entry_price <= 0:
        return None
    if has_open_ticker(acct, ticker):
        return None
    cost = round(count * entry_price, 2)
    if cost > acct.cash_usd:
        return None
    pos = PaperPosition(
        ticker=ticker,
        side=side,
        count=count,
        entry_price=entry_price,
        strike=strike,
        close_time=close_time or "",
        edge=edge,
        confidence=confidence,
        jev_p_up=jev_p_up,
        jev_action=jev_action,
        jev_model=jev_model,
        spot_at_entry=spot_at_entry,
        opened_at=_now(),
        cost_usd=cost,
    )
    acct.cash_usd = round(acct.cash_usd - cost, 2)
    acct.positions.append(pos)
    save_account(acct)
    _append_trade_log({
        "event": "open",
        "ts": pos.opened_at,
        "ticker": ticker,
        "side": side,
        "count": count,
        "entry_price": entry_price,
        "cost_usd": cost,
        "strike": strike,
        "edge": edge,
        "confidence": confidence,
        "jev_p_up": jev_p_up,
        "jev_action": jev_action,
        "jev_model": jev_model,
        "cash_after": acct.cash_usd,
        "equity_after": acct.equity_usd,
    })
    return pos


def settle_due_positions(acct: PaperAccount, settle_spot: float) -> list[PaperPosition]:
    """Settle any open position whose close_time is in the past."""
    now = datetime.now(timezone.utc)
    settled: list[PaperPosition] = []
    for pos in acct.open_positions:
        if not pos.close_time:
            continue
        try:
            close_dt = datetime.fromisoformat(pos.close_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        if close_dt > now:
            continue
        settled_up = settle_spot >= pos.strike
        won = (pos.side == "yes" and settled_up) or (pos.side == "no" and not settled_up)
        payout = pos.count * 1.0 if won else 0.0
        pnl = round(payout - pos.cost_usd, 2)
        pos.status = "settled"
        pos.settle_spot = settle_spot
        pos.settled_up = settled_up
        pos.pnl_usd = pnl
        pos.settled_at = _now()
        acct.cash_usd = round(acct.cash_usd + payout, 2)
        acct.realized_pnl_usd = round(acct.realized_pnl_usd + pnl, 2)
        acct.trades += 1
        if won:
            acct.wins += 1
        else:
            acct.losses += 1
        settled.append(pos)
        _append_trade_log({
            "event": "settle",
            "ts": pos.settled_at,
            "ticker": pos.ticker,
            "side": pos.side,
            "won": won,
            "settled_up": settled_up,
            "settle_spot": settle_spot,
            "strike": pos.strike,
            "pnl_usd": pnl,
            "cash_after": acct.cash_usd,
            "equity_after": acct.equity_usd,
            "realized_pnl_usd": acct.realized_pnl_usd,
        })
    if settled:
        save_account(acct)
    return settled


def summary(acct: PaperAccount) -> dict[str, Any]:
    wr = (acct.wins / acct.trades) if acct.trades else None
    return {
        "bankroll_usd": acct.bankroll_usd,
        "cash_usd": round(acct.cash_usd, 2),
        "equity_usd": acct.equity_usd,
        "realized_pnl_usd": round(acct.realized_pnl_usd, 2),
        "open_positions": len(acct.open_positions),
        "trades": acct.trades,
        "wins": acct.wins,
        "losses": acct.losses,
        "win_rate": round(wr, 3) if wr is not None else None,
        "state_path": str(_STATE_PATH),
        "trades_path": str(_TRADES_PATH),
    }
