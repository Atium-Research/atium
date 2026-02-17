import dataframely as dy


class Alphas(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    signal = dy.String()
    alpha = dy.Float()


class BenchmarkWeights(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    weight = dy.Float()


class Returns(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    return_ = dy.Float(alias="return")


class FactorLoadings(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    factor = dy.String()
    loading = dy.Float()


class FactorCovariances(dy.Schema):
    date = dy.Date()
    factor_1 = dy.String()
    factor_2 = dy.String()
    covariance = dy.Float()


class IdioVol(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    idio_vol = dy.Float()


class PortfolioWeights(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    weight = dy.Float()


class PositionResults(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    weight = dy.Float()
    value = dy.Float()
    return_ = dy.Float(alias="return")
    pnl = dy.Float()


class PortfolioReturns(dy.Schema):
    date = dy.Date()
    portfolio_value = dy.Float()
    portfolio_return = dy.Float()


class BenchmarkReturns(dy.Schema):
    date = dy.Date()
    benchmark_return = dy.Float()


class ActiveReturns(dy.Schema):
    date = dy.Date()
    active_return = dy.Float()


class PerformanceSummary(dy.Schema):
    annualized_return_pct = dy.Float()
    annualized_volatility_pct = dy.Float()
    sharpe_ratio = dy.Float()
    max_drawdown_pct = dy.Float()
    active_return_pct = dy.Float(nullable=True)
    tracking_error_pct = dy.Float(nullable=True)
    information_ratio = dy.Float(nullable=True)
    relative_max_drawdown_pct = dy.Float(nullable=True)