# Quick start

## 1. Define a coupled model

```python
import numpy as np
from landfeedback import CoupledModel, FeedbackAnalyzer

x_eq = np.array([0.60, 20.0])
J = np.array([[-0.20, -0.03], [-1.10, -0.50]])

def tendency(x):
    return J @ (x - x_eq)

model = CoupledModel(
    state_names=["soil_moisture", "soil_temperature"],
    tendency=tendency,
    units=["1", "degC"],
    time_unit="day",
)
```

## 2. Analyze the equilibrium

```python
result = FeedbackAnalyzer(model).analyze(
    equilibrium=x_eq,
    scales=[0.05, 2.0],
)

result.matrix_frame()
result.relaxation_modes()
```

Use `FeedbackAnalyzer(model).at_equilibrium(x0=..., scales=...)` when the
equilibrium must be solved numerically.

## 3. Evaluate an anomaly

```python
warm_dry = np.array([0.55, 22.0])
result.contribution_frame(warm_dry)
```

The returned table contains the standardized target and predictor anomalies,
scaled coefficients, actual pathway contributions, and feedback classes.

## 4. Plot the result

```python
from landfeedback.plotting import plot_feedback_matrix, plot_contributions

plot_feedback_matrix(result)
plot_contributions(result, warm_dry, target="soil_moisture")
```

