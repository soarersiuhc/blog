---
layout: post
title: "CIFAR-10 #2 - Linear classifiers: Support Vector Machine"
page_name: CIFAR-10--2---Linear-classifiers--Support-Vector-Machine
date: 2020-06-14
categories: data-science cs231n
---

In this post we will apply Support Vector Machine (SVM) to the CIFAR-10 data set.

Main references are:

- [cs231n notes](https://cs231n.github.io/linear-classify/)
- [this stackexchange answer](https://stats.stackexchange.com/questions/23391/how-does-a-support-vector-machine-svm-work)
- [bloomberg crash course on machine learning](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/04c.SVM.pdf)

## Base case: Two class SVM

#### Score and loss

Suppose we have data $(\vec{x_1}, y_1), \cdots, (\vec{x_n}, y_n)$, with $\vec{x_i}$ being input and $y_i = 0, 1$ being label (the class it belongs to). If these two classes are linearly separable, then there must be some weight vectors $\vec{w}$ s.t. 

- If $\vec{w} \cdot \vec{x} > 0$, then $y = 1$.
- If $\vec{w} \cdot \vec{x} < 0$, then $y = 0$.

We can think of $s(x) := \vec{w} \cdot \vec{x}$ as the score function. (In reality, it is really the relative score function: score for class 1 - score for class 0, as in the case of [logistic regression]({{ site.baseurl }}{% post_url 2020-06-07-CIFAR-10--2---Linear-classifiers %})).

One can then ask, which such $\vec{w}$ is the best? One natural answer would be

- when $y = 1$, $\vec{w} \cdot \vec{x}$ should be quite positive
- when $y = 0$, $\vec{w} \cdot \vec{x}$ should be quite negative. 

So we can formulate the loss function 

$$\begin{cases}\max(0, 1 - s(x)) & \text{ if label = 1} \\ \max(0, 1 + s(x)) & \text{ if label = 0}\end{cases}$$

Why do we use 1? Well, "quite positive" and "quite negative" is meaningless without fixing a scale, since we can always multiply a scalar to the score to scale up and down. So we can use an arbitrary positive number, in this case 1, as a reference point to say, $s(x) \ge 1$ is positive enough for label 1, and $s(x) \leq -1$ is negative enough for label 0.

To simplify the expression of loss, it's conventional to use $\pm 1$ as positive/negative label in this case, so that we can write 

$$\max(0, 1 - ys(x))$$

as the loss directly. For clarity, we will use $y^{\pm}$ to mean we use $\pm 1$ as class labels. This is the [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) by the way.

Anyhow, we now have 
- a score function $s(x) = \vec{w} \cdot \vec{x}$.
- a loss function to minimize,

  $$L = \sum_i \max(0, 1 - y_i^{\pm} s(x_i))$$

  One may also add a regularization term to the loss. If we think a large weight is exotic, we can add size of the weight as the loss. The $L^2$-size is often use here because of a geometric interpretation as we shall see. For example it may look like

  $$L = \frac{1}{n} \sum_i \max(0, 1 - y_i^{\pm} s(x_i)) + \lambda \|\vec{w}\|_2^2$$

  Note that we average the hinge loss, because we don't want regularization to get weaker as number of samples increase.

#### Minimizing the loss

Even though the loss is piecewise linear and not differentiable at the "hinges", gradient-based methods still work since this loss function is sub-differentiable. For example, we can apply the subgradient descent.

- Basic property of subgradient
- subgradient method [link](https://see.stanford.edu/materials/lsocoee364b/01-subgradients_notes.pdf), [link](https://web.stanford.edu/class/ee392o/subgrad_method.pdf)

#### Geometric interpretation

- Lagrangian dual [link](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture7.pdf)
- Rephrase SVM hinge loss, then dualize [link](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/04c.SVM.pdf)

## Multi-class case

## Applying SVM to CIFAR-10
