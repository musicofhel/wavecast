"""Multi-asset universe definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from wavecast.core.types import AssetClass, Sector


@dataclass
class AssetSpec:
    """Specification for a single asset."""

    ticker: str
    name: str
    asset_class: AssetClass
    sector: Sector | None = None


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

    def by_sector(self, sector: Sector) -> list[AssetSpec]:
        return [a for a in self.assets if a.sector == sector]

    def __len__(self) -> int:
        return len(self.assets)


LEGACY_UNIVERSE = Universe(
    name="legacy",
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

# Phase 3 universe: US equities + commodity ETFs with sector classification
DEFAULT_UNIVERSE = Universe(
    name="default",
    assets=[
        # Tech (5)
        AssetSpec("AAPL", "Apple", AssetClass.EQUITY, Sector.TECH),
        AssetSpec("MSFT", "Microsoft", AssetClass.EQUITY, Sector.TECH),
        AssetSpec("GOOGL", "Alphabet", AssetClass.EQUITY, Sector.TECH),
        AssetSpec("AMZN", "Amazon", AssetClass.EQUITY, Sector.TECH),
        AssetSpec("NVDA", "NVIDIA", AssetClass.EQUITY, Sector.TECH),
        # Finance (3)
        AssetSpec("JPM", "JPMorgan Chase", AssetClass.EQUITY, Sector.FINANCE),
        AssetSpec("GS", "Goldman Sachs", AssetClass.EQUITY, Sector.FINANCE),
        AssetSpec("BAC", "Bank of America", AssetClass.EQUITY, Sector.FINANCE),
        # Energy (3)
        AssetSpec("XOM", "Exxon Mobil", AssetClass.EQUITY, Sector.ENERGY),
        AssetSpec("CVX", "Chevron", AssetClass.EQUITY, Sector.ENERGY),
        AssetSpec("COP", "ConocoPhillips", AssetClass.EQUITY, Sector.ENERGY),
        # Healthcare (3)
        AssetSpec("JNJ", "Johnson & Johnson", AssetClass.EQUITY, Sector.HEALTHCARE),
        AssetSpec("UNH", "UnitedHealth", AssetClass.EQUITY, Sector.HEALTHCARE),
        AssetSpec("PFE", "Pfizer", AssetClass.EQUITY, Sector.HEALTHCARE),
        # Broad ETFs (2)
        AssetSpec("SPY", "S&P 500 ETF", AssetClass.EQUITY, Sector.BROAD_ETF),
        AssetSpec("QQQ", "Nasdaq 100 ETF", AssetClass.EQUITY, Sector.BROAD_ETF),
        # Commodity ETFs (4)
        AssetSpec("GLD", "Gold ETF", AssetClass.COMMODITY, Sector.COMMODITY_ETF),
        AssetSpec("SLV", "Silver ETF", AssetClass.COMMODITY, Sector.COMMODITY_ETF),
        AssetSpec("USO", "Oil ETF", AssetClass.COMMODITY, Sector.COMMODITY_ETF),
        AssetSpec("UNG", "Natural Gas ETF", AssetClass.COMMODITY, Sector.COMMODITY_ETF),
    ],
)

# Backward compatibility alias
PHASE3_UNIVERSE = DEFAULT_UNIVERSE


_UNIVERSES: dict[str, Universe] = {
    "default": DEFAULT_UNIVERSE,
    "phase3": PHASE3_UNIVERSE,
    "legacy": LEGACY_UNIVERSE,
}


def get_universe(name: str = "default") -> Universe:
    """Get a named universe."""
    if name not in _UNIVERSES:
        raise ValueError(
            f"Unknown universe: {name}. Available: {list(_UNIVERSES.keys())}"
        )
    return _UNIVERSES[name]
