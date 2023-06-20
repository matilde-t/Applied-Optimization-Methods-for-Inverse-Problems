# Homework 5

## Part 1: Proximal operators

I decided to put all these proximal operators into a class, that acts as a collection so that they don't get confused with their respective functions. Furthermore, for generalization purposes the function should all have the same signature, even if they require different parameters. Therefore, I put the parameter setup in the class declaration to make the call of these functions easier. The class is implemented in `ProximalOperators.py`.

Here are the plots of these operators in the interval $[-10, 10]$, with various values of sigma when applicable.

![](constant.png)

![](huber.png)

![](l2.png)

![](translation.png)

## Part 2: Proximal Gradient Method

### i) Proximal Gradient Method

For this task, I took inspiration from `IterativeShrinkageThresholdingAlgorithm.py` and generalized it in `ProximalGradientMehod.py`.

Here are some results with different proximal operators.

![](proximal-Constant-backtracking.png)

![](proximal-Huber-backtracking.png)

![](proximal-L2-backtracking.png)

I tried to use line search, but the results aren't great. Same goes with BB1 and BB2.

![](proximal-Constant-BB2.png)

![](proximal-Huber-BB2.png)

![](proximal-L2-BB2.png)

The results don't seem great: I obtain an oscillatory behaviour while using BB1/BB2, and it doesn't converge well with backtracking. I tried different values of $\sigma$ without any significant improvement.

### ii) Fast Proximal Gradient Method

I decided to add this feature directly into the PGM class with two flags, `fast1` and `fast2`.

![](proximal-Constant-fast1.png)

![](proximal-Huber-fast1.png)

![](proximal-L2-fast1.png)

![](proximal-Constant-fast2.png)

![](proximal-Huber-fast2.png)

![](proximal-L2-fast2.png)

As we can see, it doesn't help much. This time, I didn't use any search for the value of $\lambda$.

### iii) Uniqueness of formulation

In the case of $g(x)=\frac{1}{2}||Ax-b||_2^2+\frac{\beta}{2}||x||_2^2$ and $h(x)=0$, we can work with the canonical gradient descent with Tikhonov regularization.

In the case of $g(x)=\frac{1}{2}||Ax-b||_2^2$ and $h(x)=\frac{\beta}{2}||x||_2^2$ instead, we can use PGM with the proximal operator of the L2 norm.

![](conv-GD.png)

![](conv-PGM.png)

We can see that the PGM method converges quickly, but stalls after a short time, whereas GD converges more slowly but gets closer to the ground truth.

### iv) Elastic Net Formulation

## Part 3: Restart Conditions
