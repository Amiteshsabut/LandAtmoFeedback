from __future__ import annotations

import numpy as np

from landfeedback.benchmarks import brubaker1996


def test_state_statistics_and_parameters():
    states = brubaker1996.state_statistics().set_index("variable")
    assert states.loc["soil_moisture", "equilibrium"] == 0.613
    assert states.loc["soil_temperature", "stationary_sd"] == 2.1
    parameters = brubaker1996.main_parameters().set_index("symbol")
    assert parameters.loc["h", "value"] == 1000.0
    assert parameters.loc["U_bar", "value"] == 4.0


def test_published_reporting_matrix():
    matrix = brubaker1996.published_reporting_matrix()
    assert matrix.shape == (2, 5)
    assert matrix.loc["soil_moisture_tendency", "soil_moisture"] == -0.64
    assert matrix.loc["soil_temperature_tendency", "soil_temperature"] == -3.81
    assert matrix.loc["soil_temperature_tendency", "delta_temperature"] == -2.91


def test_displayed_process_sums_match_published_rounding():
    moisture = brubaker1996.deterministic_soil_moisture()
    a11 = moisture[(moisture.term == "A11") & (moisture.process != "sum")]
    assert np.isclose(a11.scaled_value_mm_day.sum(), -0.64, atol=0.011)

    temperature = brubaker1996.deterministic_soil_temperature()
    for term in ["A31", "A32", "A33", "A34", "A35"]:
        rows = temperature[temperature.term == term]
        displayed_sum = rows.loc[rows.process == "sum", "scaled_value_degC_day"].iloc[0]
        component_sum = rows.loc[rows.process != "sum", "scaled_value_degC_day"].sum()
        assert np.isclose(component_sum, displayed_sum, atol=0.011)


def test_conditional_and_stochastic_fixtures():
    moisture = brubaker1996.conditional_soil_moisture().set_index("component")
    assert moisture.loc["linearized_tendency", "dry_mm_day"] == 0.21
    assert moisture.loc["nonlinear_tendency", "moist_mm_day"] == -0.16
    temperature = brubaker1996.conditional_soil_temperature().set_index("component")
    assert temperature.loc["nonlinear_tendency", "dry_degC_day"] == -2.64
    stochastic = brubaker1996.stochastic_soil_temperature().set_index("term")
    assert stochastic.loc["Lambda33", "scaled_value_degC_day"] == -0.53

