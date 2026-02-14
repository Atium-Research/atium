"""
Harbinger API Design Sketch
============================

This is a pseudo-Python file illustrating the desired syntax.
Not meant to run - just to nail down the API before implementing.
"""

from harbinger import (
    # Core
    Backtester,
    DataAdapter,
    TradingConfig,
    
    # Objectives
    Objective,
    MaximizeAlpha,
    MaximizeSharpe,
    MinimizeVariance,
    MinimizeActiveVariance,
    TargetActiveRisk,
    MinimizeTurnover,
    RiskParity,
    
    # Costs
    TransactionCost,
    SlippageModel,
)

# Optimizer constraints (convex, go into the QP)
from harbinger.optimizer_constraints import (
    LongOnly,
    FullyInvested,
    MaxWeight,
    MaxSectorWeight,
    MaxFactorExposure,
)

# Trading constraints (post-optimization heuristics)
from harbinger.trading_constraints import (
    MinTradeSize,
    MaxTurnover,
    RoundLots,
)


# =============================================================================
# 1. DATA ADAPTER
# =============================================================================
# Users implement this to connect their data to harbinger's expected format.
# Harbinger calls these methods during backtest.

class MyDataAdapter(DataAdapter):
    """User implements this to provide their data."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    # Required methods
    def get_universe(self, date) -> list[str]:
        """Return list of tickers in universe for this date."""
        return self.db.query_universe(date)
    
    def get_alphas(self, date) -> dict[str, float]:
        """Return expected returns / alpha scores for this date."""
        return self.db.query_alphas(date)
    
    def get_covariance(self, date) -> tuple[np.ndarray, list[str]]:
        """Return (cov_matrix, tickers) for this date."""
        return self.db.query_covariance(date)
    
    def get_prices(self, date) -> dict[str, float]:
        """Return prices for this date."""
        return self.db.query_prices(date)
    
    def get_returns(self, date) -> dict[str, float]:
        """Return realized returns for this date (for PnL calculation)."""
        return self.db.query_returns(date)
    
    # Optional methods (required by some objectives/constraints)
    def get_benchmark_weights(self, date) -> dict[str, float]:
        """Required if using TargetActiveRisk or MinimizeActiveVariance."""
        return self.db.query_benchmark(date)
    
    def get_sector_mapping(self, date) -> dict[str, str]:
        """Required if using MaxSectorWeight."""
        return self.db.query_sectors(date)
    
    def get_factor_exposures(self, date) -> dict[str, dict[str, float]]:
        """Required if using MaxFactorExposure. Returns {ticker: {factor: exposure}}."""
        return self.db.query_factor_exposures(date)


# =============================================================================
# 2. OBJECTIVE FUNCTION
# =============================================================================
# Composable objective terms. Each term can have parameters.
# The optimizer solves: maximize(sum of objective terms) subject to constraints

# Option A: Simple - pick one objective type
objective = TargetActiveRisk(target=0.05)  # Dynamically finds gamma

# Option B: Composite - combine multiple terms with weights
objective = Objective(
    terms=[
        MaximizeAlpha(weight=1.0),
        MinimizeActiveVariance(weight=50.0),  # gamma = 50
        MinimizeTurnover(weight=0.01),        # Penalize turnover
    ]
)

# Option C: Constrained objective - target a risk level
objective = Objective(
    maximize=MaximizeAlpha(),
    subject_to=TargetActiveRisk(target=0.05),  # This adjusts gamma dynamically
)


# =============================================================================
# 3. OPTIMIZER CONFIG
# =============================================================================
# Constraints that go into the QP solver (must be convex)

optimizer_constraints = [
    LongOnly(),
    FullyInvested(),
    MaxWeight(0.05),                          # No position > 5%
    MaxSectorWeight(0.20),                    # No sector > 20%
    MaxFactorExposure("momentum", 0.3),       # Factor exposure limits
]


# =============================================================================
# 4. TRADING CONFIG
# =============================================================================
# Post-optimization adjustments and execution parameters

trading_config = TradingConfig(
    constraints=[
        MinTradeSize(dollars=1000),           # Don't trade tiny amounts
        MaxTurnover(0.20),                    # Max 20% turnover per rebalance
        RoundLots(100),                       # Round to 100 share lots
    ],
    costs=TransactionCost(
        commission_bps=1.0,
        slippage=SlippageModel.SQRT_IMPACT(impact_coef=0.1),
    ),
    rebalance_frequency="weekly",             # or "daily", "monthly", custom
)


# =============================================================================
# 5. BACKTESTER
# =============================================================================

backtest = Backtester(
    data=MyDataAdapter(db),
    objective=objective,
    optimizer_constraints=optimizer_constraints,
    trading=trading_config,
    initial_capital=1_000_000,
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# Run it
result = backtest.run()

# Results
result.print_summary()
result.plot_equity_curve()
result.to_dataframe()  # Daily results


# =============================================================================
# 6. DEPENDENCY RESOLUTION
# =============================================================================
# Some configurations require certain data. The backtester validates this
# before running and gives clear error messages.

# Example: This would fail validation
bad_config = Backtester(
    data=AdapterWithoutBenchmark(),           # Doesn't implement get_benchmark_weights
    objective=TargetActiveRisk(target=0.05),  # Requires benchmark!
)
# Raises: DependencyError: "TargetActiveRisk requires DataAdapter.get_benchmark_weights()"

# The resolver checks:
# - TargetActiveRisk / MinimizeActiveVariance → needs get_benchmark_weights
# - MaxSectorWeight → needs get_sector_mapping
# - MaxFactorExposure → needs get_factor_exposures
# - etc.


# =============================================================================
# 7. ALTERNATIVE: FLUENT BUILDER API
# =============================================================================
# Some people prefer this style:

result = (
    Backtester(MyDataAdapter(db))
    .with_objective(TargetActiveRisk(0.05))
    .with_constraints([
        LongOnly(),
        FullyInvested(),
        MaxWeight(0.05),
    ])
    .with_trading(
        min_trade=1000,
        max_turnover=0.20,
        costs=TransactionCost(commission_bps=1.0),
    )
    .run(
        start="2020-01-01",
        end="2024-12-31",
        capital=1_000_000,
    )
)


# =============================================================================
# EXAMPLE SETUPS
# =============================================================================

# -----------------------------------------------------------------------------
# Setup 1: Target Active Risk (Long-Only Equity)
# -----------------------------------------------------------------------------
# Goal: Maximize alpha while targeting specific tracking error vs benchmark
# Use case: Active equity fund benchmarked to S&P 500

setup_1 = Backtester(
    data=EquityDataAdapter(db),
    objective=Objective(
        maximize=MaximizeAlpha(),
        subject_to=TargetActiveRisk(target=0.05),  # 5% tracking error
    ),
    optimizer_constraints=[
        LongOnly(),
        FullyInvested(),
    ],
    trading=TradingConfig(
        constraints=[
            MinPositionSize(dollars=10_000),
        ],
        costs=TransactionCost(commission_bps=1.0, slippage_bps=5.0),
    ),
)


# -----------------------------------------------------------------------------
# Setup 2: Long-Short Market Neutral
# -----------------------------------------------------------------------------
# Goal: Portable alpha with zero market exposure
# Use case: Hedge fund stat-arb strategy

setup_2 = Backtester(
    data=EquityDataAdapter(db),
    objective=Objective(
        terms=[
            MaximizeAlpha(weight=1.0),
            MinimizeVariance(weight=10.0),  # Fixed gamma = 10
        ]
    ),
    optimizer_constraints=[
        DollarNeutral(),                    # Long $ = Short $
        SectorNeutral(),                    # Neutral within each sector
        MaxWeight(0.05),                    # Max 5% long
        MinWeight(-0.05),                   # Max 5% short
        MaxGrossExposure(2.0),              # 200% gross (100L + 100S)
    ],
    trading=TradingConfig(
        constraints=[
            MaxTurnover(0.30),              # Max 30% turnover per rebal
            MinTradeSize(dollars=5_000),
        ],
        costs=TransactionCost(
            commission_bps=1.0,
            slippage_bps=10.0,              # Higher for shorts
            borrow_cost_bps=50.0,           # Cost to borrow shorts
        ),
    ),
)


# -----------------------------------------------------------------------------
# Setup 3: Minimum Variance (No Alpha)
# -----------------------------------------------------------------------------
# Goal: Lowest volatility portfolio
# Use case: Defensive allocation, risk-off regime

setup_3 = Backtester(
    data=EquityDataAdapter(db),
    objective=MinimizeVariance(),           # No alpha term at all
    optimizer_constraints=[
        LongOnly(),
        FullyInvested(),
        MaxWeight(0.05),                    # Diversification
    ],
    trading=TradingConfig(
        constraints=[
            MinTradeSize(dollars=1_000),
            RoundLots(shares=100),
        ],
        costs=TransactionCost(commission_bps=1.0),
        rebalance_frequency="monthly",      # Less frequent for min-var
    ),
)


# -----------------------------------------------------------------------------
# Setup 4: Max Sharpe Ratio (Tangency Portfolio)
# -----------------------------------------------------------------------------
# Goal: Maximize risk-adjusted returns, no benchmark
# Use case: Absolute return fund, no tracking error concern

setup_4 = Backtester(
    data=EquityDataAdapter(db),
    objective=MaximizeSharpe(),             # Tangency portfolio
    optimizer_constraints=[
        LongOnly(),
        FullyInvested(),
        MaxWeight(0.10),
    ],
    trading=TradingConfig(
        constraints=[
            MinTradeSize(dollars=5_000),
            MaxTurnover(0.25),
        ],
        costs=TransactionCost(commission_bps=1.0, slippage_bps=5.0),
    ),
)


# -----------------------------------------------------------------------------
# Setup 5: Risk Parity
# -----------------------------------------------------------------------------
# Goal: Equal risk contribution from each asset
# Use case: Multi-asset allocation

setup_5 = Backtester(
    data=MultiAssetDataAdapter(db),
    objective=RiskParity(),                 # Special objective type
    optimizer_constraints=[
        LongOnly(),
        FullyInvested(),
    ],
    trading=TradingConfig(
        constraints=[
            MaxTurnover(0.10),
        ],
        costs=TransactionCost(commission_bps=2.0),
        rebalance_frequency="monthly",
    ),
)


# =============================================================================
# OPEN QUESTIONS
# =============================================================================
"""
1. Should objective terms be additive or should we have a clear hierarchy?
   - Additive: maximize(alpha - gamma * variance - penalty * turnover)
   - Hierarchical: maximize alpha subject to risk <= target

2. How to handle the gamma vs target_active_risk duality?
   - Option A: Two separate objective types (MinimizeVariance vs TargetActiveRisk)
   - Option B: One objective with mode parameter
   - Option C: TargetActiveRisk wraps another objective and adjusts it

3. Should optimizer constraints be part of the objective or separate?
   - Mathematically they're different (objective vs constraint)
   - But users might think of them together

4. Rebalance frequency - how flexible?
   - Fixed: daily/weekly/monthly
   - Custom: user provides list of dates
   - Conditional: rebalance when turnover > threshold

5. How to handle missing data?
   - Skip the day?
   - Use last available?
   - Require adapter to handle it?
"""
