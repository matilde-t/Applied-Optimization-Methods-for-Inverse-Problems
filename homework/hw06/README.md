# Homework 6

## Part 1: Proximal Optimized Gradient Method

For this assignment, I build yet another class on the style of the previous ones, in  `ProximalOptimizedGradientMethod.py`. I had to modify the class in `ProximalOperators.py` in order to accomodate the varying $\sigma$ value required by the algorithm, and I implemented the non-negativity constraint on all the used methods. I make a comparison among OGM1, PGM and OGM, with varying parameters. I used BB1 (when applicable) and the results seem quite satisfying. I'm using the phantom 07A with a 90° angle.

![](1_recon_PGM_constant.png)

![](1_recon_PGM_huber.png)

![](1_recon_PGM_l2.png)

![](1_recon_POGM_constant.png)

![](1_recon_POGM_huber.png)

![](1_recon_POGM_l2.png)

![](1_recon_OGM.png)

![](1_scores.png)

I decided to plot directly the scores. As we can see, some algorithms "get stuck" more or less at the same level: OGM seems to reach the POGM level, but slowly.

Here are the final scores, side by side:

- OGM: 0.7632
- PGM, constant operator: 0.8281
- PGM, Huber operator: 0.4042
- PGM, L2 operator: 0.4165
- POGM, constant operator: **0.8581**
- POGM, Huber operator: 0.4110
- POGM, L2 operator: 0.7211

In bold the best score.

## Part 2: Expectation Maximization

We did this homework in class. You can see the results under `lesson/XLFM_exercise`.

## Part 3: Low Dose X-ray CT

I put the provided snippets into `LowDoseRead.py` file, taking inspiration from `challenge/utils.py` in order to directly obtain the sinogram and XrayOperator. I needed to bin the sinogram because when I tried to backproject I was going out of memory: to compensate for this, I consider every second angle in the angles list. I also return an initial filtered backprojection, to see what I am doing.

![](2_full_dose_0.png)

![](2_full_dose_10.png)

![](2_full_dose_20.png)

![](2_full_dose_30.png)

![](2_low_dose_0.png)

![](2_low_dose_10.png)

![](2_low_dose_20.png)

![](2_low_dose_30.png)

We can see that the difference between the low-dose and full-dose data is hardly noticeable. The relative error for the slices is in this table, for a simple backprojection with Ram-Lak filter.

| Slice | Relative error |
| :---: | :---: |
| 0  | 8.67% |
| 10 | 6.87% |
| 20 | 7.16% |
| 30 | 7.72% |

Probably the relative  error is a bit higher in the first slice due to the black bands resulting from the conversion from helical trajectories with a curved detector to circular trajectories with a flat detector.

For the reconstruction I decided to concentrate on the slice 10, given that it seems the best-behaved one. In oder not to have scale problems, here I use the score as a metric. The score of the filtered backprojection, using the full dose's filtered backprojection as the ground truth, is already quite good, with a value of **0.9857**. It is quite hard to do better: here I tried a couple of methods, but every one achieves a lower score.

![](2_scores_LS.png)

With the log likelihood I get comparable results. I had to set $I_0$ to 1, instead of $1e^8$ as written in the assignment.

![](2_scores_LL.png)

Given that I had to use adaptive step sizes with least squares and I'm not using them with the log likelihood and the results are comparable, I can say that probably the log likelihood works better.
