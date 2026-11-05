"""
Test that orchestrator and stock-oracle prompts do not contain pressure to
override HOLD or force trades after streaks.
"""
from crypto_oracle.orchestrator import _SYNTH_SYSTEM as CRYPTO_SYSTEM
from crypto_oracle.stock_oracle import _SYNTH_SYSTEM as STOCK_SYSTEM


def test_crypto_prompt_has_no_prime_directive_pressure():
    """The old PRIME DIRECTIVE forced trades over HOLDs — must be gone."""
    assert "PRIME DIRECTIVE" not in CRYPTO_SYSTEM
    assert "HOLD is almost always wrong" not in CRYPTO_SYSTEM
    assert "Sitting on cash generates nothing" not in CRYPTO_SYSTEM
    assert "Default to BUY or SELL" not in CRYPTO_SYSTEM


def test_crypto_prompt_allows_hold():
    """HOLD must be presented as a normal, valid option."""
    assert "HOLD" in CRYPTO_SYSTEM
    assert "valid" in CRYPTO_SYSTEM.lower()


def test_crypto_prompt_no_threshold_ceiling_at_065():
    """Old ceiling was 0.65 to force trades — threshold must allow higher values."""
    # The prompt should not instruct to keep threshold ≤ 0.65
    assert "never raise above 0.65" not in CRYPTO_SYSTEM
    assert "0.48" not in CRYPTO_SYSTEM  # old floor used to encourage overtrading


def test_stock_prompt_has_no_streak_size_pressure():
    """Old prompt pressured increasing size on winning streaks."""
    assert "increase size" not in STOCK_SYSTEM.lower().replace("Adjust by", "adj")
    assert "Increase by $50" not in STOCK_SYSTEM
    assert "winning streak" not in STOCK_SYSTEM


def test_stock_prompt_hold_is_valid():
    """HOLD must be presented as a normal and valid response."""
    assert "always a valid" in STOCK_SYSTEM.lower() or "valid" in STOCK_SYSTEM.lower()


def test_crypto_orchestrator_threshold_allows_90_pct():
    """confidence_threshold can now reach 0.90 (old cap was 0.65)."""
    # Import the actual clamp applied by orchestrator
    from crypto_oracle.orchestrator import CryptoOracle
    import inspect
    src = inspect.getsource(CryptoOracle._synthesise if hasattr(CryptoOracle, '_synthesise') else CryptoOracle)
    # The code must allow max up to 0.90
    assert "0.90" in src or "0.9" in src
