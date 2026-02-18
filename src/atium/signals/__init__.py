import dataframely as dy
import polars as pl

from atium.schemas import AlphasSchema, ScoresSchema
from atium.types import IdioVol, Scores, Signals, Universe


def compute_scores(signals: Signals, name: str) -> Scores:
    return ScoresSchema.validate(
        signals
        .with_columns(
            pl.col('signal')
            .sub(pl.col('signal').mean())
            .truediv(pl.col('signal').std())
            .over('date')
            .alias('score'),
            pl.lit(name).alias('name')
        )
        .select('date', 'ticker', 'name', 'score')
    )

def compute_alphas(universe: Universe, scores: Scores, idio_vol: IdioVol, name: str) -> dy.DataFrame[AlphasSchema]:
    return AlphasSchema.validate(
        universe
        .join(scores, on=['date', 'ticker'], how='left')
        .join(idio_vol, on=['date', 'ticker'], how='left')
        .with_columns(
            pl.lit(.05)
            .mul(pl.col('score'))
            .mul(pl.col('idio_vol'))
            .fill_null(0)
            .alias('alpha'),
            pl.lit(name).alias('name')
        )
        .select('date', 'ticker', 'name', 'alpha')
    )
