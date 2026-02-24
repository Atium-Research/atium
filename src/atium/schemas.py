import dataframely as dy


class SignalsSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    signal = dy.Float(nullable=True)


class ScoresSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    score = dy.Float(nullable=True)


class AlphasSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    alpha = dy.Float(nullable=True)


class UniverseSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()


class BenchmarkWeightsSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    weight = dy.Float()


class ReturnsSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    return_ = dy.Float(alias="return")


class FactorLoadingsSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    factor = dy.String()
    loading = dy.Float()


class FactorCovariancesSchema(dy.Schema):
    date = dy.Date()
    factor_1 = dy.String()
    factor_2 = dy.String()
    covariance = dy.Float()


class IdioVolSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    idio_vol = dy.Float()


class BetasSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    beta = dy.Float()


class PortfolioWeightsSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    weight = dy.Float()


class PositionResultsSchema(dy.Schema):
    date = dy.Date()
    ticker = dy.String()
    weight = dy.Float()
    value = dy.Float()
    return_ = dy.Float(alias="return")
    pnl = dy.Float()


