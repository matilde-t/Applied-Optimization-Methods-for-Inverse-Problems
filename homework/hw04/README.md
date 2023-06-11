# Homework 4

## Part 1: Even more gradient based methods

### i) Choosing the correct step length

I modify the `GradientDescentClass.py` in order to include these two line search algorithms through some boolean flags. For testing, I re-use the same test functions I used in *Homework 2*, selected from Wikipedia's [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization). I directly confront backtracking, BB1 and BB2 at the same time.

#### Booth function

$$ f(x,y)=(x+2y-7)^2+(2x+y-5)^2$$
$$\nabla f(x,y)=[10x+8y-34,\quad 8x+10y-38]$$

![Booth function](booth.png "Booth function")

I highlighted the point of minimum, that is $(1,3)$.

With starting point $(-5,-5)$, this is the performance for different line search algorithms:

| Algorithm  | $x$  | Number of iterations | Absolute error |
|:---:|:---:|:---:|:---:|
| Backtracking  | $(1.00000267, 2.99999745)$  | $36$ | $3.6911\cdot10^{-6}$ |
| Barzilai and Borwein 1  | $(1., 3.)$  | $8$ | $6.2804\cdot10^{-16}$ |
| Barzilai and Borwein 2  | $(1., 3.)$  | $8$ | $1.1102\cdot10^{-16}$ |

We can see that in particular BB1 and BB2 achieve a great result, with machine precision and a very low number of iterations.

#### Three-hump camel function

$$f(x,y)=2x^2-1.05x^4+\frac{x^6}{6}+xy+y^2$$
$$\nabla f(x,y)=[4x-4.2x^3+x^5+y, \quad x+2y]$$

![Camel function](camel.png "Camel function")

I highlighted the point of minimum, that is $(0,0)$.

With starting point $(-5,-5)$, this is the performance for different line search algorithms:

| Algorithm  | $x$  | Number of iterations | Absolute error |
|:---:|:---:|:---:|:---:|
| Backtracking  | $(-1.8687\cdot10^{-8},\\ -2.6356\cdot10^{-7})$  | $21$ | $2.6422\cdot10^{-7}$ |
| Barzilai and Borwein 1  | $(2.3530\cdot10^{-11},\\ 9.3548\cdot10^{-12})$  | $19$ | $2.5322\cdot10^{-11}$ |
| Barzilai and Borwein 2  | $(-2.3139\cdot10^{-11},\\ -9.5845\cdot10^{-12})$  | $18$ | $2.5046\cdot10^{-11}$ |

In this case, the number of iterations is comparable in the three line search algorithms, but BB1 and BB2 achieve a greater precision.

#### Helsinki tomography dataset

In this case, I use the 07 C sample from the dataset. In this case, instead of the absolute error, it is more relevant to calculate the correlation score with the given tool.

![](htc2022_orig.png)

What I found out is that these methods require a lower number of iterations and actually perform worse with a higer one: for example with BB2 the score was becoming negative with 1000 iterations.

![](htc2022_Line_Search.png)

![](htc2022_Barzilai_and_Borwein_1.png)

![](htc2022_Barzilai_and_Borwein_2.png)

We can see that in the case of backtracking, the value of $\lambda$ has a stable, oscillating pattern, whereas BB1 and BB2 follow a similar pattern with occasional "spikes". Probably in the case of BB2 these "spikes" are too big for this problem and cause the algorithm to acutally go away from the solution on the long run.

![](htc2022_convergence.png)

From this last plot, we can see that Backtracking is more stable, but decreases more slowly compared to BB1. BB2 also performs quite well, but it's more unstable.

### ii) Iterative Shrinkage-Thresholding Algorithm

For the creation of this algorithm, in the file `IterativeShrinkageThresholdingAlgorithm.py`, I followed the same structure as `GradientDescentClass.py` and compared the performance of the algorithm with a set $\beta=10^{-2}$. As we can see, a dynamic step size helps immensely and we observe a behaviour similar to the one in point i).

![](ISTA_Default.png)

![](ISTA_Line_Search.png)

![](ISTA_Barzilai_and_Borwein_1.png)

![](ISTA_Barzilai_and_Borwein_2.png)

![](ISTA_convergence.png)

### iii) Projected Gradient Descent

I implemented this function in `ProjectedGradientDescent.py` following the same scheme as `IterativeShrinkageThresholdingAlgorithm.py`. I try to use the non-negativity projection and test some upper bounds. I observe the convergence speed and the score obtained after 100 iterations (I found out that they are enough to show the behaviour).To make the challenge more interesting, I use the usual 07 C sample from the dataset, but with a 90° angle.

![](PDG_convergence_-inf-50.png)
![](PDG_final_-inf-50.png)

![](PDG_convergence_-inf-150.png)
![](PDG_final_-inf-150.png)

![](PDG_convergence_-inf-250.png)
![](PDG_final_-inf-250.png)

![](PDG_convergence_-inf-inf.png)
![](PDG_final_-inf-inf.png)

![](PDG_convergence_0-50.png)
![](PDG_final_0-50.png)

![](PDG_convergence_0-150.png)
![](PDG_final_0-150.png)

![](PDG_convergence_0-250.png)
![](PDG_final_0-250.png)

![](PDG_convergence_0-inf.png)
![](PDG_final_0-inf.png)

To recap these results, in this table we can see the scores of the various methods for a specific value interval:

| Interval  | Default | Backtracking | BB1 | BB2 |
|:---:|:---:|:---:|:---:|:---:|
| $(-\infty,50]$ | $0.4179$ | $0.8781$ | $0.8871$ | $0.8878$ |
| $(-\infty,150]$ | $0.4179$ | $0.8781$ | $0.8871$ | $0.8878$ |
| $(-\infty,250]$ | $0.4179$ | $0.8781$ | $0.8871$ | $0.8878$ |
| $(-\infty,\infty)$ | $0.4179$ | $0.8781$ | $0.8871$ | $0.8878$ |
| $[0,50]$ | $0.4184$ | $0.9167$ | $0.9221$ | $0.9328$ |
| $[0,150]$ | $0.4184$ | $0.9167$ | $0.9221$ | $0.9328$ |
| $[0,250]$ | $0.4184$ | $0.9167$ | $0.9221$ | $0.9328$ |
| $[0,\infty)$ | $0.4184$ | $0.9167$ | $0.9221$ | $0.9328$ |

As we can see, the upper bound doesn't have an impact, but the non-negativity factor helps a lot.

## Part 2: Semi-Convergence

## Part 3: Challenge Dataset

This time, I try to use the image 07 A, with 90°, 60° and 30° angles.

I tried with backtracking, BB1 and BB2, and discovered that the best results were with BB2, so I will show those here.

For ISTA, I simply restrict the negative, values, because as we've seen it's the most important improvement.

These are the results for 1000 iterations:

| Angle | Gradient Descent | ISTA | Projected Gradient Descent |
|:---:|:---:|:---:|:---:|
| $90°$ | $0.8134$ | $\boldsymbol{0.8771}$ | $0.8714$ |
| $60°$ | $0.7058$ | $\boldsymbol{0.8216}$ | $0.8098$ |
| $30°$ | $0.5763$ | $0.7457$ | $\boldsymbol{0.7471}$ |

In bold the best scores. Unsurprisingly, a wider arc provides an easier reconstruction, and it seems that the regularization properties of ISTA and the negative values restricion help in improving the performance.

![](htc2022_07a_GD.png)

![](htc2022_07a_ISTA.png)

![](htc2022_07a_PGD.png)
