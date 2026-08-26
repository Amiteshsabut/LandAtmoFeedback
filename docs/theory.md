# Theory

## Coupled tendency model

Let the deterministic model be

\[
\frac{d\mathbf{x}}{dt}=\mathbf{G}(\mathbf{x}),
\]

and let \(\mathbf{x}^*\) satisfy \(\mathbf{G}(\mathbf{x}^*)=0\). The local
linearization is

\[
\mathbf{G}(\mathbf{x})\approx
J(\mathbf{x}-\mathbf{x}^*),
\]

where

\[
J_{ij}=\left.\frac{\partial G_i}{\partial x_j}\right|_{\mathbf{x}^*}.
\]

## Scaling

With a positive scale \(\sigma_j\), normally a stationary standard deviation,

\[
\delta x_j=\frac{x_j-x_j^*}{\sigma_j},
\qquad
A_{ij}=J_{ij}\sigma_j.
\]

The scaled matrix \(A\) expresses every predictor perturbation in a comparable
one-scale unit. The tendency units remain those of the target variable per
model time unit.

## Contributions and feedback class

At a specified state,

\[
C_{ij}=A_{ij}\delta x_j.
\]

The pathway restores target \(i\) when

\[
C_{ij}\delta x_i<0,
\]

and reinforces it when the product is positive. A zero target anomaly or zero
contribution is classified as neutral.

This means an off-diagonal coefficient does not have a universal feedback
class. Its physical role depends on the covariance structure and the current
anomaly signs.

## Stability and relaxation

The equilibrium is locally asymptotically stable when every eigenvalue of
\(J\) has a negative real part. For a stable mode with eigenvalue \(\lambda\),
the local e-folding time is

\[
\tau=-\frac{1}{\Re(\lambda)}.
\]

These are local results. They need not describe recovery far from equilibrium
when the full tendency is strongly nonlinear.

## Process decomposition

If

\[
\mathbf{G}=\sum_p\mathbf{G}^{(p)},
\]

then

\[
J=\sum_p J^{(p)}.
\]

The package differentiates each named process independently. The
`process_closure` residual checks whether their Jacobians sum to the total.

## Stochastic extension

For

\[
d\mathbf{x}=\mathbf{G}(\mathbf{x})dt+\mathbf{g}(\mathbf{x})d\mathbf{W},
\]

`landfeedback` evaluates \(\mathbf{g}(\mathbf{x}^*)\) and its scaled state
sensitivity. Simulation uses Euler-Maruyama, so users must test timestep
convergence for their model.

## Derived-variable warning

If a reported quantity is algebraically derived from other states, it cannot be
added as an independent numerical Jacobian coordinate without a coordinate
transformation. For example, Brubaker and Entekhabi report soil temperature,
air temperature, and their difference. The difference must be handled through
independent coordinates or named process grouping to avoid double counting.

