---
layout: post
title: Roth's theorem 1 - different proofs
date: 2020-05-26
categories: arithmetic-combinatorics, szemeredi's theorem, math
---

**Theorem (Roth, infinitary)** A subset of $\mathbb{N}$ with positive natural upper density contains a 3-term arithmetic progression (3-AP).

By a compactness argument, this is equivalent to a finitary statement:

**Theorem (Roth, finitary)** Given any $\delta > 0$, we can find $N(\delta)$, such that if $N > N(\delta)$, then any subset $A \subset \mathbb{Z}/N\mathbb{Z}$ with $|A| > \delta N$ contains a 3-term arithmetic progression (3-AP).

We will outline a few proofs of the finitary theorem.

# Proof 1: Density increment

Consider the sum
$$S_A := \sum_{x, d \in \mathbb{Z}/N\mathbb{Z}} 1_A(x) 1_A(x+d) 1_A(x+2d)$$
This captures 3-APs that lie in $A$, including the trivial ones where $d = 0$. If $A$ has density $\delta$, then on average we can pretend $1_A$ is $\delta \cdot 1$, and see $S_A = \delta^3 N^2$. This means on average we expect there are many 3-term APs, since the trivial ones only account for $\delta N$ of them.

Our first proof would proceed as follows:

1. either $A$ is "random" enough, so that $S_A \approx \delta^3 N^2$ and we know there are many 3-APs
2. or that $A$ will correlate with some character of $\mathbb{Z}/N\mathbb{Z}$, which allows us to find a sub-AP of $\mathbb{Z}/N\mathbb{Z}$ with size $N^{1/3}$, on which $A$ has increased density > $\delta + c \delta^6$.
This process would then need to stop in finitely many steps, since the density cannot be larger than 1. So case 1 must happen at some point, and so there must be an AP!

A key thing here is to capture "randomness" - one way to phrase it is $1_A - \delta \cdot 1$ is "small" in some sense. In this case we can use "Fourier coefficients are small on average" to quantify this, but to make this generalizable to $k$-term AP case, we will use the language of Gowers norm to capture this randomness.

## Digression: Gowers $U^2$ norm
For real-valued $f \in L^2(\mathbb{Z}/N\mathbb{Z})$, we define the Gowers $U^2$ norm by
$$\|f\|_{U^2}^{2^2} = \mathbb{E}_{x,d_1,d_2 \in \mathbb{Z}/N\mathbb{Z}} f(x) f(x+d_1) f(x+d_2) f(x+d_1+d_2)$$

- $U^2$-norm is a norm. In fact,
$$\|f\|_{U^2}^{2^2} = \sum_{\chi \in \mathbb{Z}/N\mathbb{Z}^*} |\hat{f}(\chi)|^4$$
This also implies that
$$\|f\|_{U^2} \leq \|\hat{f}\|_{\infty}^{1/2}$$
since
$$\|f\|_{U^2}^4 \leq \|\hat{f}\|_{\infty}^2 \sum_{\chi} |\hat{f}(\chi)|^2 = \|\hat{f}\|_{\infty}^2 \|f\|_{L^2}^2 \leq \|\hat{f}\|_{\infty}^2$$
by Plancherel identity.
- (von-Neumann theorem) If $f, g, h: \mathbb{Z}/N\mathbb{Z} \to [-1, 1]$, and define $$\Lambda_3 (f, g, h) := \mathbb{E}_{x, d} f(x) g(x+d) h(x + 2d)$$
then $$|\Lambda_3 (f, g, h)| \leq \inf \{ \|f\|, \|g\|, \|h\| \}$$

## Density increment
Consider $$S := \mathbb{E}_{x, d \in \mathbb{Z}/N\mathbb{Z}} 1_A(x) 1_A(x+d) 1_A(x+2d).$$
By writing $1_A = \delta \cdot 1 + (1_A - \delta \cdot 1)$, and using von Neumann's theorem, we see that
$$S = \delta^3 + O(\|1_A - \delta \cdot 1\|_{U^2}) $$

- If $A$ is "random", manifested as $$\|1_A - \delta \cdot 1\|_{U^2} \ll \delta^3,$$ then $S \gg \delta^3$, meaning there are already many AP's.
- Otherwise $\|1_A - \delta \cdot 1\|_{U^2} \gg \delta^3$, and hence by inverse theorem $$\|1_A - \delta \cdot 1\|_{\infty} \gg \delta^6.$$
