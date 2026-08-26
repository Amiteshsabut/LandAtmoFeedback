from __future__ import annotations

import pytest

from landfeedback.cli import main


def test_cli_prints_benchmark(capsys):
    assert main(["benchmark", "reporting-matrix"]) == 0
    output = capsys.readouterr().out
    assert "soil_moisture_tendency" in output
    assert "-3.81" in output


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out

