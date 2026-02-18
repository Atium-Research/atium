from typing import TypeAlias

import dataframely as dy

from atium.schemas import (ActiveReturnsSchema, AlphasSchema,
                           BenchmarkReturnsSchema, BenchmarkWeightsSchema,
                           FactorCovariancesSchema, FactorLoadingsSchema,
                           IdioVolSchema, PerformanceSummarySchema,
                           PortfolioReturnsSchema, PortfolioWeightsSchema,
                           PositionResultsSchema, ReturnsSchema, ScoresSchema,
                           SignalsSchema, UniverseSchema)

Signals: TypeAlias = dy.DataFrame[SignalsSchema]
Scores: TypeAlias = dy.DataFrame[ScoresSchema]
Universe: TypeAlias = dy.DataFrame[UniverseSchema]
Alphas: TypeAlias = dy.DataFrame[AlphasSchema]
BenchmarkWeights: TypeAlias = dy.DataFrame[BenchmarkWeightsSchema]
Returns: TypeAlias = dy.DataFrame[ReturnsSchema]
FactorLoadings: TypeAlias = dy.DataFrame[FactorLoadingsSchema]
FactorCovariances: TypeAlias = dy.DataFrame[FactorCovariancesSchema]
IdioVol: TypeAlias = dy.DataFrame[IdioVolSchema]
PortfolioWeights: TypeAlias = dy.DataFrame[PortfolioWeightsSchema]
PositionResults: TypeAlias = dy.DataFrame[PositionResultsSchema]
PortfolioReturns: TypeAlias = dy.DataFrame[PortfolioReturnsSchema]
BenchmarkReturns: TypeAlias = dy.DataFrame[BenchmarkReturnsSchema]
ActiveReturns: TypeAlias = dy.DataFrame[ActiveReturnsSchema]
PerformanceSummary: TypeAlias = dy.DataFrame[PerformanceSummarySchema]
