"""Multi-asset universe definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from wavecast.core.types import AssetClass


@dataclass
class AssetSpec:
    """Specification for a single asset."""

    ticker: str
    name: str
    asset_class: AssetClass


@dataclass
class Universe:
    """A collection of assets to analyze."""

    name: str
    assets: list[AssetSpec] = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    def by_class(self, cls: AssetClass) -> list[AssetSpec]:
        return [a for a in self.assets if a.asset_class == cls]

    def __len__(self) -> int:
        return len(self.assets)


DEFAULT_UNIVERSE = Universe(
    name="default",
    assets=[
        # Equities (8)
        AssetSpec("AAPL", "Apple", AssetClass.EQUITY),
        AssetSpec("MSFT", "Microsoft", AssetClass.EQUITY),
        AssetSpec("GOOGL", "Alphabet", AssetClass.EQUITY),
        AssetSpec("AMZN", "Amazon", AssetClass.EQUITY),
        AssetSpec("JPM", "JPMorgan Chase", AssetClass.EQUITY),
        AssetSpec("XOM", "Exxon Mobil", AssetClass.EQUITY),
        AssetSpec("SPY", "S&P 500 ETF", AssetClass.EQUITY),
        AssetSpec("QQQ", "Nasdaq 100 ETF", AssetClass.EQUITY),
        # Crypto (4)
        AssetSpec("X:BTCUSD", "Bitcoin", AssetClass.CRYPTO),
        AssetSpec("X:ETHUSD", "Ethereum", AssetClass.CRYPTO),
        AssetSpec("X:SOLUSD", "Solana", AssetClass.CRYPTO),
        AssetSpec("X:XRPUSD", "XRP", AssetClass.CRYPTO),
        # Forex (3)
        AssetSpec("C:EURUSD", "Euro/Dollar", AssetClass.FOREX),
        AssetSpec("C:GBPUSD", "Pound/Dollar", AssetClass.FOREX),
        AssetSpec("C:USDJPY", "Dollar/Yen", AssetClass.FOREX),
        # Commodity ETFs (4)
        AssetSpec("GLD", "Gold ETF", AssetClass.COMMODITY),
        AssetSpec("SLV", "Silver ETF", AssetClass.COMMODITY),
        AssetSpec("USO", "Oil ETF", AssetClass.COMMODITY),
        AssetSpec("UNG", "Natural Gas ETF", AssetClass.COMMODITY),
    ],
)


_UNIVERSES: dict[str, Universe] = {
    "default": DEFAULT_UNIVERSE,
}


def get_universe(name: str = "default") -> Universe:
    """Get a named universe."""
    if name not in _UNIVERSES:
        raise ValueError(
            f"Unknown universe: {name}. Available: {list(_UNIVERSES.keys())}"
        )
    return _UNIVERSES[name]
