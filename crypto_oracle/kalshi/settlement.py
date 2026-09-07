"""Pure settlement helpers shared by paper and live calibration paths."""
from __future__ import annotations


def yes_outcome_at_settlement(
    settle_price: float, strike: float, cap_strike: float | None = None
) -> bool:
    """Return whether YES wins for a directional threshold or range contract."""
    if cap_strike is not None:
        return strike <= settle_price < cap_strike
    return settle_price >= strike