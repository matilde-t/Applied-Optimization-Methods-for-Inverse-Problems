# Homework 3

In order to improve ease of use, I decided to refactor `GradientDescent.py` in an object oriented way. I kept as reference the old version and implemented the new one in `GradientDescentClass.py`.

## Part 1: More gradient based methods

### i) Nesterov's Methods

I followed the same principle as the new `GradientDescentClass.py` file and implemented OGM1 as a class. As an initial image value, I use the one obtained from the reconstruction performed with the inverse radon transform and the Ram-Lak filter. Then, I try to apply Gradient Descent and Optimized Gradient Method to the classical least squares problem and the one with L2-regularization.

![](htc2022_orig.png)

The results seem quite promising, especially with L2-regularization. It seems that the method, compared to Gradient Descent, tends to overshooting. I had to reduce $\lambda$ to $10^{-4}$ in order to achieve similar results.

![](htc2022_comp.png)

### ii) Landweber iteration

