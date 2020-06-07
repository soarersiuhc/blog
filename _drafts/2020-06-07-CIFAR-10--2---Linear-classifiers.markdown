---
layout: post
title: "CIFAR-10 #2 - Linear classifiers"
page_name: CIFAR-10--2---Linear-classifiers
date: 2020-06-07
categories: data-science cs231n
---

In this post we will try linear classifiers to tell the images apart. We will try a few methods here:
- Multinomial logistic regression/Softmax
- Support Vector Machine (SVM)
- Linear Discriminant Analysis (LDA)

Of course, all of these have the assumption that our input data is "separable". Logistic regression/SVM relies on the data being (almost) separable by a hyperplane. LDA is somewhat different, but effectively assuming each class clusters around a center, and is as if it's sampled from a Gaussian distribution at the center. 

It doesn't seem like people verify these assumptions in practice; a priori for arbitrary raw input, I am not convinced that this is generally the case (although empirical it seems to work fairly well). This goes back to the problem of finding good embeddings of the inputs. By composing with a "good" kernel if necessary, it is fair to say a "good" embedding should have some linear separability built in. This is perhaps why in many neural network architectures, the last layer is usually a softmax layer - one can pretend that all the previous layers try to learn a "good" embedding so that things can be linearly separated.

**Remark** Another desired linear property of "good" embedding is that dot product of embedding is meaningful. This is not something we need here, but it's another major thing I don't understand and ideally have some heuristic justification at least.

Main reference is [cs231n notes](https://cs231n.github.io/linear-classify/)

## Model framework

To build a model, we will follow the framework of
- finding a suitable score function
- use a suitable loss function
- use regularization techniques to reduce overfitting.

For linear classifiers, the score function we use is an affine linear function of the inputs, hence *linear*.

## Multinomial logistic regression/Softmax

### Two class case: logistic regression

Suppose we want to predict a binary output $y = 0$ or 1. We may model $\mathbb{P}(y = 1)$, and predict $y = 1$ whenever $\mathbb{P}(y = 1)$ is larger than some threshold. This is equivalent to model $\mathbb{P}(y = 0)$ when there are two classes, since $\mathbb{P}(y = 0) + \mathbb{P}(y = 1) = 1$. 

Suppose we want to use a linear model for this. One problem is that an affine linear function is valued in $(-\infty, \infty)$, while probability should lie in $[0,1]$. To resolve this, one can rescale $(-\infty, \infty)$ to $(0,1)$ by a nice, increasing function (so that $-\infty \to 0$, $\infty \to 1$) , i.e. take some $g: (-\infty, \infty) \to (0,1)$, and assert that 

$$\mathbb{P}(y = 1) = g(a_1 x_1 + \cdots + a_kx_k)$$

if $(x_1, \cdots, x_k)$ is the input. If $g$ is invertible - then we are postulating $g^{-1} (\mathbb{P}(y = 1))$ should be linear:

$$g^{-1}(\mathbb{P}(y = 1)) = a_1x_1 + \cdots + a_kx_k$$

Common choices of $g$ are:
- Expit function $g(x) = \frac{1}{1 + e^{-x}}$. Its inverse is the logit function 

$$g^{-1}(p) = logit(p) = \log\left(\frac{p}{1-p}\right).$$

We can interpret this as the log odds: 

$$\log \frac{\mathbb{P}(y = 1)}{\mathbb{P}(y = 0)}.$$ 

From the inference perspective, this allows us to say something like: if an input increases by 1 unit, the odds of a positive label will increase by x times. This is the logistic regression.

- CDF of standard Gaussian $g(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{x^2}{2}} dx$. Its inverse is the probit function, and this is called the probit regression.

- Log-Weibull distribution $g(x) = 1 - e^{-e^{x}}$. Its inverse is the complementary log-log function

$$g^{-1}(p) = \log(-\log(1-p))$$

**Question** When to use which link function?

### Questions
- Why is centering/normalizing input needed, given that they all have the same range [0, 255]? Try to not normalize.
