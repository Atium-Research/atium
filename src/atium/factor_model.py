import polars as pl
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from atium.schemas import (FactorCovariancesSchema, FactorLoadingsSchema,
                           IdioVolSchema)
from atium.types import FactorCovariances, FactorLoadings, IdioVol, Returns


def estimate_factor_model(
    stock_returns: Returns,
    factor_returns: Returns,
    window: int = 252,
    half_life: int = 60,
) -> tuple[FactorLoadings, IdioVol]:
    factors = factor_returns["ticker"].unique().sort().to_list()

    df = stock_returns.join(
        other=factor_returns.pivot(on="ticker", index="date", values="return"),
        how="left",
        on="date",
    )

    results = []
    for ticker in df["ticker"].unique():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")

        if len(ticker_df) < window:
            continue

        y = ticker_df["return"].to_pandas()
        X = sm.add_constant(ticker_df.select(factors).to_pandas())

        rolling_model = RollingOLS(y, X, window=window).fit()

        result = pl.DataFrame(
            {
                "ticker": ticker,
                "date": ticker_df["date"],
                "return": ticker_df["return"],
                "alpha": rolling_model.params["const"],
            }
            | {factor: ticker_df[factor] for factor in factors}
            | {f"B_{factor}": rolling_model.params[factor] for factor in factors}
        ).with_columns(
            pl.col("return")
            .sub(
                pl.col("alpha")
                + pl.sum_horizontal(
                    pl.col(factor).mul(pl.col(f"B_{factor}"))
                    for factor in factors
                )
            )
            .alias("residual")
        )
        results.append(result)

    results = pl.concat(results)

    factor_loadings = FactorLoadingsSchema.validate(
        results.select("ticker", "date", *[f"B_{factor}" for factor in factors])
        .unpivot(
            index=["ticker", "date"], variable_name="factor", value_name="loading"
        )
        .with_columns(
            pl.col("factor").str.replace(r"^B_", ""),
            pl.col("loading").ewm_mean(half_life=half_life).over("ticker", "factor"),
        )
        .drop_nulls("loading")
        .select("date", "ticker", "factor", "loading")
    )

    idio_vol = IdioVolSchema.validate(
        results.sort("ticker", "date")
        .select(
            "date",
            "ticker",
            pl.col("residual")
            .rolling_std(window_size=window)
            .ewm_mean(half_life=half_life)
            .over("ticker")
            .alias("idio_vol"),
        )
        .drop_nulls("idio_vol")
    )

    return factor_loadings, idio_vol


def estimate_factor_covariances(
    factor_returns: Returns,
    window: int = 252,
    half_life: int = 60,
) -> FactorCovariances:
    factor_returns_pd = (
        factor_returns.sort("ticker", "date")
        .pivot(on="ticker", index="date", values="return")
        .to_pandas()
        .set_index("date")
    )

    factor_covariances = (
        pl.from_pandas(
            factor_returns_pd.rolling(window=window, min_periods=window)
            .cov()
            .reset_index()
        )
        .rename({"level_1": "factor_1"})
        .with_columns(pl.col("date").dt.date())
    )

    return FactorCovariancesSchema.validate(
        factor_covariances.drop_nulls()
        .unpivot(
            index=["date", "factor_1"],
            variable_name="factor_2",
            value_name="covariance",
        )
        .sort("factor_1", "factor_2", "date")
        .with_columns(
            pl.col("covariance").ewm_mean(half_life=half_life).over("factor_1", "factor_2"),
        )
        .select("date", "factor_1", "factor_2", "covariance")
    )
