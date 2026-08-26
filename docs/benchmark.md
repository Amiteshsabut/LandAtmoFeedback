# Brubaker and Entekhabi (1996) benchmark

The package includes the published equilibrium statistics, main parameter
values, deterministic process decompositions, conditional dry/moist
contributions, and perturbation-forcing sensitivities from Tables 2-9.

```python
from landfeedback.benchmarks import brubaker1996

brubaker1996.state_statistics()
brubaker1996.main_parameters()
brubaker1996.deterministic_soil_moisture()
brubaker1996.deterministic_soil_temperature()
brubaker1996.published_reporting_matrix()
brubaker1996.conditional_soil_moisture()
brubaker1996.conditional_soil_temperature()
brubaker1996.stochastic_soil_moisture()
brubaker1996.stochastic_soil_temperature()
```

The CLI provides the same data:

```bash
landfeedback benchmark all
```

## Validation status

The values are transcription-checked regression fixtures. Several process
subtotals in the article differ by 0.01 from sums of displayed components due
to published rounding.

The package does not yet claim exact end-to-end reproduction of the four-state
physical model. The analysis article explicitly refers to:

- Brubaker and Entekhabi (1995), DOI `10.1029/94WR01772`;
- Entekhabi and Brubaker (1995), DOI `10.1029/94WR01773`.

Those sources are required to establish all construction and stochastic-model
details before an exact reproduction can be validated.

