import numpy as np

from landfeedback import ReportingDecomposition, full_validation_report
from landfeedback.benchmarks import brubaker1996


def test_full_validation_suite_passes():
    report = full_validation_report()
    print(report.summary())
    assert report.passed


def test_paper_reporting_matrix_shape_and_values():
    matrix = brubaker1996.published_reporting_matrix()
    assert matrix.shape == (2, 5)
    assert matrix.loc["soil_moisture_tendency", "soil_moisture"] == -0.64
    assert matrix.loc["soil_temperature_tendency", "soil_temperature"] == -3.81
    assert matrix.loc["soil_temperature_tendency", "delta_temperature"] == -2.91


def test_paper_table_arithmetic_is_consistent_with_rounding():
    frame = brubaker1996.paper_arithmetic_checks()
    assert frame["absolute_difference"].max() <= 0.03


def test_reporting_decomposition_supports_derived_channel():
    decomp = brubaker1996.reporting_decomposition()
    assert isinstance(decomp, ReportingDecomposition)
    assert "delta_temperature" in decomp.predictor_names
    delta = np.array([-1.0, 1.0, 1.0, 1.0, 1.0])
    contributions = decomp.contributions(delta)
    assert contributions.shape == (2, 5)
