# Homework 7

## Part 1: More Proximal Operators

I add these two operators to `ProximalOperators.py`.

## Part 2: ADMM

I implement this algorithm in `LinearADMM.py`, with the usual layout of my other algorithms. I directly implement it with the XrayOperator in mind. I test it with phantom 7b and 60 degree angle.

### i) LASSO Problem

As we can see, there is basically no difference for different values of $\tau$, and the reconstruction isn't the best. These are the results after 100 iterations.

![](2_tau_1e-03.png)

![](2_tau_1e-02.png)

![](2_tau_1e-01.png)

![](2_tau_1e+00.png)

![](2_tau_1e+01.png)

![](2_tau_1e+02.png)

![](2_tau_1e+03.png)

![](2_tau_1e+04.png)

![](2_tau_1e+05.png)

![](2_tau_1e+06.png)

### ii) TV regularization



## Part 3: Challenge data
