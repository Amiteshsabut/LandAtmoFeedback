# LandAtmoFeedback

![LandAtmoFeedback banner](assets/landatmo-feedback-banner.png)

`landfeedback` calculates local feedback matrices, pathway contributions,
process attribution, dry/wet composites, stability modes, and stochastic
sensitivity for coupled dynamical models.

The software separates three ideas that are often conflated:

1. A **Jacobian derivative** describes local dynamical sensitivity.
2. A **scaled coefficient** describes the effect of a one-standard-deviation
   predictor anomaly.
3. A **feedback classification** depends on the current predictor and target
   anomalies and says whether the pathway restores or reinforces the target.

Start with the [quick-start example](quickstart.md), then read the
[theory](theory.md) before interpreting off-diagonal feedbacks.

## Scope

The core package is model-agnostic. Published values from Brubaker and
Entekhabi (1996) are bundled as benchmark tables. They are not presented as a
complete reproduction of the physical model because two companion articles
contain additional construction and parameterization details.
