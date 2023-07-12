# Homework 8

## Part 1: Subgradient Method

I implement this method in `SubgradientMethod.py`, taking inspiration from my Gradient Descent class. 

![](1_SGM_constant.png)

![](1_SGM_decreasing.png)

![](1_SGM_constant_err.png)

![](1_SGM_decreasing_err.png)

I tried with different $\beta$ values and with a constant or decreasing step size.

With a constant step size, the best $\beta$ value seems to be around $10^{-3}$ or $10^{-2}$.

With a decreasing step size we obtain a similar result, but the reconstruction is stuck at a score of approximately 0.38.

I now changed the ADMM implementation, given that I realized that I set the wrong values of the $\sigma$ parameter of the proximal operators. Now there is a visible difference sometimes, but score-wise the result is still far from optimal.

Here a couple of examples, with isotropic TV regularization:

![](2_tau_1e-02_iso.png)

![](2_tau_1e+00_iso.png)

![](2_tau_1e+02_iso.png)

![](2_tau_1e+04_iso.png)

![](2_tau_1e+06_iso.png)

