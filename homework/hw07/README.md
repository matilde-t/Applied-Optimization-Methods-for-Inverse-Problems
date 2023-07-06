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

#### Stacked operators

I implemented this functionality in a class called `stack` inside `LinearADMM.py`, given that it will mostly be used there. I used the `.apply()` and `.applyAdjoint()` logic to be consistent with `XrayOperator` and `FirstDerivative`, which is implemented based on the *Appendix* in file `ForwardDifferences.py`.

#### Separable sum

I also implemented these two features directly inside `LinearADMM.py`, as two classes that take the list of functions and the input vector.

#### Reconstructions

The reconstructions I get aren't very good and oddly enough I tried with a lot of different parameter combinations and the algorithm seems super unsensitive to them, I get the exact same result every time.

##### Isotropic

![](3_tau_1e-03_iso.png)

![](3_tau_1e-02_iso.png)

![](3_tau_1e-01_iso.png)

![](3_tau_1e+00_iso.png)

![](3_tau_1e+01_iso.png)

![](3_tau_1e+02_iso.png)

![](3_tau_1e+03_iso.png)

![](3_tau_1e+04_iso.png)

![](3_tau_1e+05_iso.png)

![](3_tau_1e+06_iso.png)

##### Anisotropic

![](3_tau_1e-03_aniso.png)

![](3_tau_1e-02_aniso.png)

![](3_tau_1e-01_aniso.png)

![](3_tau_1e+00_aniso.png)

![](3_tau_1e+01_aniso.png)

![](3_tau_1e+02_aniso.png)

![](3_tau_1e+03_aniso.png)

![](3_tau_1e+04_aniso.png)

![](3_tau_1e+05_aniso.png)

![](3_tau_1e+06_aniso.png)
