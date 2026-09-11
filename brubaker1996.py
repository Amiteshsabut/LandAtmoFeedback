"""Published regression fixtures from Brubaker & Entekhabi (1996).

The paper's analysis combines a four-state nonlinear stochastic model with a
local linearization.  The 1996 paper refers to two 1995 companion papers for
additional model-construction/parameterization details, so the tables below are
treated as published-value validation fixtures rather than a claim of a fully
independent reconstruction of the complete physical model.

Important: ``delta_temperature = soil_temperature - air_temperature`` is a
derived reporting channel, not a fifth prognostic state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..decomposition import ReportingDecomposition


def citation() -> str:
    return (
        "Brubaker, K. L., & Entekhabi, D. (1996). Analysis of feedback "
        "mechanisms in land-atmosphere interaction. Water Resources Research, "
        "32(5), 1343-1357. https://doi.org/10.1029/96WR00005"
    )


def methodology_summary() -> tuple[str, ...]:
    """Concise section-3/6/7 workflow implemented by LandAtmoFeedback."""
    return (
        "Solve or supply the deterministic equilibrium x* where G(x*)=0.",
        "Linearize the deterministic drift G around x*.",
        "Scale each predictor derivative by its stationary standard deviation: "
        "A_ij=(dG_i/dx_j)|x* sigma_j.",
        "Decompose A_ij into physical process contributions where available.",
        "Use a stochastic joint state distribution to obtain physically consistent "
        "conditional disequilibria at the dry (<=5th percentile) and moist "
        "(>=95th percentile) soil-moisture tails.",
        "Multiply A_ij by conditional expected standardized disequilibria and sum "
        "to obtain linearized conditional tendencies; compare with conditional "
        "means of the nonlinear G.",
        "Linearize the stochastic susceptibility g around x*: "
        "g_i(x)≈g_i(x*)+sum_j Lambda_ij delta_j, with "
        "Lambda_ij=(dg_i/dx_j)|x* sigma_j.",
    )


def state_statistics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("soil_moisture", "1", 0.613, 0.611, 0.043),
            ("air_humidity", "g kg-1", 4.27, 4.25, 0.39),
            ("soil_temperature", "degC", 20.7, 20.6, 2.1),
            ("air_temperature", "degC", 15.3, 15.5, 1.4),
            ("delta_temperature", "degC", 5.4, 5.3, 1.7),
        ],
        columns=["variable", "unit", "equilibrium", "stationary_mean", "stationary_sd"],
    )


def main_parameters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("h", "mixed-layer height", 1000.0, "m"),
            ("p_s", "surface pressure", 1000.0, "mbar"),
            ("p_h", "mixed-layer-top pressure", 880.0, "mbar"),
            ("A_top", "sensible-heat entrainment parameter", 0.2, "1"),
            ("b", "moistening partition parameter", 0.3, "1"),
            ("Z_h", "hydrologically active soil depth", 0.2, "m"),
            ("Z_t", "thermally active soil depth", 0.4, "m"),
            ("n", "soil porosity", 0.25, "1"),
            ("U_bar", "mean mixed-layer wind speed", 4.0, "m s-1"),
            ("sigma_u", "wind-speed standard deviation", 1.5, "m s-1"),
            ("L", "regional length scale", 500.0, "km"),
            ("q_in", "incoming-air specific humidity", 8.0, "g kg-1"),
        ],
        columns=["symbol", "parameter", "value", "unit"],
    )


def deterministic_soil_moisture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A11", "soil_moisture", "advected precipitation infiltration", -0.25),
            ("A11", "soil_moisture", "recycled precipitation", -0.03),
            ("A11", "soil_moisture", "evaporation efficiency", -0.36),
            ("A11", "soil_moisture", "sum", -0.64),
            ("A12", "air_humidity", "vapor gradient in potential evaporation", 0.08),
            ("A13", "soil_temperature", "saturation specific humidity", -0.48),
            ("A14", "air_temperature", "buoyancy velocity", 0.00),
            ("A15", "delta_temperature", "buoyancy velocity", -0.22),
        ],
        columns=["term", "predictor", "process", "scaled_value_mm_day"],
    )


def deterministic_soil_temperature() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A31", "soil_moisture", "albedo", 0.11),
            ("A31", "soil_moisture", "evaporation efficiency", -0.99),
            ("A31", "soil_moisture", "sum", -0.88),
            ("A32", "air_humidity", "shortwave cloud correction", -1.02),
            ("A32", "air_humidity", "longwave from above mixed layer", 0.15),
            ("A32", "air_humidity", "column absorption", -0.21),
            ("A32", "air_humidity", "clear-sky longwave from mixed layer", 0.34),
            ("A32", "air_humidity", "cloud longwave correction", 0.17),
            ("A32", "air_humidity", "vapor deficit in potential evaporation", 0.50),
            ("A32", "air_humidity", "sum", -0.08),
            ("A33", "soil_temperature", "longwave from soil", -1.19),
            ("A33", "soil_temperature", "saturation specific humidity", -2.62),
            ("A33", "soil_temperature", "sum", -3.81),
            ("A34", "air_temperature", "shortwave cloud correction", 0.44),
            ("A34", "air_temperature", "above-layer downwelling longwave", 0.09),
            ("A34", "air_temperature", "layer downwelling longwave", 0.35),
            ("A34", "air_temperature", "buoyancy velocity in sensible heat", 0.00),
            ("A34", "air_temperature", "buoyancy velocity in evaporation", 0.02),
            ("A34", "air_temperature", "sum", 0.90),
            ("A35", "delta_temperature", "gradient in sensible heat", -1.52),
            ("A35", "delta_temperature", "buoyancy velocity in sensible heat", -0.35),
            ("A35", "delta_temperature", "buoyancy velocity in evaporation", -1.05),
            ("A35", "delta_temperature", "sum", -2.91),
        ],
        columns=["term", "predictor", "process", "scaled_value_degC_day"],
    )


def published_reporting_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [-0.64, 0.08, -0.48, 0.00, -0.22],
            [-0.88, -0.08, -3.81, 0.90, -2.91],
        ],
        index=pd.Index(
            ["soil_moisture_tendency", "soil_temperature_tendency"], name="target"
        ),
        columns=pd.Index(
            [
                "soil_moisture",
                "air_humidity",
                "soil_temperature",
                "air_temperature",
                "delta_temperature",
            ],
            name="reporting_predictor",
        ),
    )


def reporting_decomposition() -> ReportingDecomposition:
    frame = published_reporting_matrix()
    return ReportingDecomposition(
        target_names=tuple(frame.index),
        predictor_names=tuple(frame.columns),
        coefficients=frame.to_numpy(float),
    )


def conditional_soil_moisture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("soil_moisture", 0.74, -0.67),
            ("air_humidity", 0.12, -0.10),
            ("soil_temperature", -0.58, 0.54),
            ("air_temperature", 0.00, 0.00),
            ("delta_temperature", -0.07, 0.08),
            ("linearized_tendency", 0.21, -0.15),
            ("nonlinear_tendency", 0.19, -0.16),
        ],
        columns=["component", "dry_mm_day", "moist_mm_day"],
    )


def conditional_soil_temperature() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("soil_moisture", 1.02, -0.93),
            ("air_humidity", -0.09, 0.07),
            ("soil_temperature", -4.15, 3.90),
            ("air_temperature", 1.12, -1.00),
            ("delta_temperature", -0.75, 0.81),
            ("linearized_tendency", -2.85, 2.87),
            ("nonlinear_tendency", -2.64, 2.32),
        ],
        columns=["component", "dry_degC_day", "moist_degC_day"],
    )


def stochastic_soil_moisture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("g1_equilibrium", "equilibrium", "equilibrium susceptibility", 0.52),
            ("Lambda11", "soil_moisture", "sum", -0.17),
            ("Lambda12", "air_humidity", "vapor gradient", 0.02),
            ("Lambda13", "soil_temperature", "saturation specific humidity", -0.11),
            ("Lambda14", "air_temperature", "does not appear", 0.00),
            ("Lambda15", "delta_temperature", "does not appear", 0.00),
        ],
        columns=["term", "predictor", "process", "scaled_value_mm_day"],
    )


def stochastic_soil_temperature() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("g3_equilibrium", "equilibrium", "equilibrium susceptibility", -3.81),
            ("Lambda31", "soil_moisture", "sum", -0.12),
            ("Lambda32", "air_humidity", "vapor deficit", 0.10),
            ("Lambda33", "soil_temperature", "saturation specific humidity", -0.53),
            ("Lambda34", "air_temperature", "does not appear", 0.00),
            ("Lambda35", "delta_temperature", "sensible-heat gradient", -0.31),
        ],
        columns=["term", "predictor", "process", "scaled_value_degC_day"],
    )


def paper_arithmetic_checks() -> pd.DataFrame:
    """Internal arithmetic checks that can be made from Tables 4-9 alone."""

    rows: list[tuple[str, float, float]] = []

    sm = deterministic_soil_moisture()
    a11_components = sm[(sm.term == "A11") & (sm.process != "sum")]
    a11_sum = float(a11_components.scaled_value_mm_day.sum())
    rows.append(("Table 4 A11 component sum", a11_sum, -0.64))

    st = deterministic_soil_temperature()
    for term in ["A31", "A32", "A33", "A34", "A35"]:
        group = st[st.term == term]
        components = group[group.process != "sum"].scaled_value_degC_day.sum()
        published = group[group.process == "sum"].scaled_value_degC_day.iloc[0]
        rows.append((f"Table 5 {term} component sum", float(components), float(published)))

    c1 = conditional_soil_moisture().set_index("component")
    rows.append(
        (
            "Table 6 dry contribution sum",
            float(c1.loc[
                ["soil_moisture", "air_humidity", "soil_temperature",
                 "air_temperature", "delta_temperature"],
                "dry_mm_day"
            ].sum()),
            float(c1.loc["linearized_tendency", "dry_mm_day"]),
        )
    )
    rows.append(
        (
            "Table 6 moist contribution sum",
            float(c1.loc[
                ["soil_moisture", "air_humidity", "soil_temperature",
                 "air_temperature", "delta_temperature"],
                "moist_mm_day"
            ].sum()),
            float(c1.loc["linearized_tendency", "moist_mm_day"]),
        )
    )

    c3 = conditional_soil_temperature().set_index("component")
    rows.append(
        (
            "Table 7 dry contribution sum",
            float(c3.loc[
                ["soil_moisture", "air_humidity", "soil_temperature",
                 "air_temperature", "delta_temperature"],
                "dry_degC_day"
            ].sum()),
            float(c3.loc["linearized_tendency", "dry_degC_day"]),
        )
    )
    rows.append(
        (
            "Table 7 moist contribution sum",
            float(c3.loc[
                ["soil_moisture", "air_humidity", "soil_temperature",
                 "air_temperature", "delta_temperature"],
                "moist_degC_day"
            ].sum()),
            float(c3.loc["linearized_tendency", "moist_degC_day"]),
        )
    )

    frame = pd.DataFrame(rows, columns=["check", "calculated", "published"])
    frame["absolute_difference"] = np.abs(frame["calculated"] - frame["published"])
    return frame
