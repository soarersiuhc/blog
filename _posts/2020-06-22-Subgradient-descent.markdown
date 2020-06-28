---
layout: post
title: "Subgradient descent"
page_name: Subgradient-descent
date: 2020-06-27
categories: data-science blurbs
---

In this post, we will go over basic properties of subgradients, as well as discussing subgradient descent.

Main references: 
- Boyd's notes [1](https://see.stanford.edu/materials/lsocoee364b/01-subgradients_notes.pdf), [2](https://web.stanford.edu/class/ee392o/subgrad_method.pdf).
- [Bloomberg crash course on machine learning](https://davidrosenberg.github.io/mlcourse/Archive/2018/Labs/4-Subgradients-Notes_sol.pdf)
- [Notes on differentiability](http://www.mat.unimi.it/users/libor/AnConvessa/differentiability.pdf)
- https://maunamn.wordpress.com/
- https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec6.pdf

---

## Subgradients: definition

**Definition (subgradient)** Let $S \subset \mathbb{R}^n$ be a convex open subset, and Let $f: S \to \mathbb{R}$ be a convex function. We say $g \in \mathbb{R}^n$ is a subgradient of $f$ at $x$ if 

$$f(z) \ge f(x) + g^T (z-x)$$

for all $z \in S$. We denote $\partial f(x)$ as the set of subgradients of $f$ at $x$.

**Definition (subdifferentiable)** $f$ is subdifferentiable at $x$ if $\partial f(x) \neq \emptyset$.

**Remark** Note that subgradient is asking the linear approximation $f(x) + g^T(z - x)$ to be a **global** minimizer. For our purpose (subgradient descent), this doesn't seem necessary: it should be sufficient that $f(z) \ge f(x) + g^T(z-x)$ true in a neighborhood around $x$, with this neighborhood uniform as $x$ varies. However,

- When we cook up new subdifferentiable function, there's more bookkeeping to do if we consider this local uniformity as well.
- If we only consider convex functions, which is what we restrict to here, then globalness and localness are equivalent.

Maybe this is why all sources I have seen so far require the minimizer to be global.

**Remark** We can also deal with more general functions:

- Our domain here is open, because this is the cleanest case. If we allow boundary points, then we generally have to be more careful there. For example, convexity does not imply continuity at the boundary points.
- One can also develop a theory for non-convex functions in general. For the sake of subgradient descent however, we basically only want to work with everywhere-subdifferentiable functions, [which is forced to be convex](https://math.stackexchange.com/questions/1499059/show-that-a-real-valued-function-with-non-empty-subdifferential-is-convex).

## Subgradients: properties

### Detour: basic properties of convex functions

Let $S \subset \mathbb{R}^n$ be a convex open subset, and Let $f: S \to \mathbb{R}$ be a convex function.

- $f$ is continuous. In fact $f$ is even locally Lipschitz.
- At each point $a \in S$ and each direction $v \in \mathbb{R}^n$, the "half" directional derivatives
  
  $$Df_a^+(v) = \lim_{h \to 0^+} \frac{f(a + hv) - f(a)}{h}, \, Df_a^-(v) = \lim_{h \to 0^-} \frac{f(a + hv) - f(a)}{h}$$

  exist and are finite.
    - This reduces to single variable case, and directly follows from convexity.
- **Lemma** $Df_a^+$ is sublinear, $Df_a^-$ is superlinear.
  - **Proof** This follows from convexity.
- $f$ is differentiable at $a$ iff directional derivatives in all directions exist.
  - **Proof (converse)** By convexity, $f$ is locally Lipschitz at $a$. Assume for the sake of contradiction that $f$ is not differentiable at $a$. Since directional derivatives all exist, there is only one possible candidate for the differential - call that $g$. Then we can find $\\|v_n\\| = 1$, $h_n \to 0^+$ such that 

    $$\frac{f(a + h_nv_n) - f(a) - h_n g^T v_n}{h_n} \not\to 0$$

    By compactness of unit ball, suppose $v_n \to v_0$. Now $f$ locally Lipschitz means that these two expressions are arbitrarily close:

    $$\frac{f(a + h_nv_n) - f(a) - h_n g^T v_n}{h_n} \sim \frac{f(a + h_nv_0) - f(a) - h_n g^T v_0}{h_n}$$

    In particular, this means $\frac{f(a + h_nv_0) - f(a) - h_n g^T v_0}{h_n} \not\to 0$, contradicting $f$ is differentiable in the direction of $v_0$. (Here implicitly I am relying on the directional derivative being linear. This is generally false for subgradients, but when the directional derivative actually exists this follows from the last bullet point via sandwich bound.)

### Properties of subgradients

- $\partial f(x)$ is a closed, non-empty, bounded convex set.
  - **Proof of closedness/convexity**: intersection of closed hyperplanes.
  - **Proof of non-emptiness**: Let $a \in S$. Consider the epigraph 
  
    $$E(f) = \{(x,y) \in S \times \mathbb{R}: y \ge f(x)\}$$

    This is convex, and $(a, f(a))$ is on the boundary of $E(f)$. By the [supporting hyperplane theorem](https://en.wikipedia.org/wiki/Supporting_hyperplane), we can find $(v, w) \neq 0$ so that

    $$(v, w)^T ((x,y) - (a, f(a))) \ge 0$$

    for all $x \in S$ with $y \ge f(x)$. This means 

    $$v^T(x-a) + w(y - f(a)) \ge 0$$

    By pushing $y \to \infty$, we see that $w \ge 0$. If $w = 0$, then $w^T(x - a) \ge 0$ for all $x \in S$ - but this is impossible since $x-a$ can cover all directions as $a$ is an interior point.

    So $w > 0$. Take $y = f(x)$, we see that 

    $$f(x) \ge f(a) - \frac{1}{w} v^T(x-a)$$

    so $- \frac{1}{w} v$ is a subgradient of $f$ at $a$.

  - **Proof of boundedness**: Let $a \in B \subset S$, where $B$ is a compact closed ball around $a$ of radius $r$. Let $M$ be the maximum of $f$ over $B$.
 
    For any subgradient $g \in \partial f(a)$, consider $x = a + r \frac{g}{\|g\|}$, then

    $$f(x) \ge f(a) + g^T (x-a) = f(a) + r \|g\| \Rightarrow \|g\| \leq \frac{f(x) - f(a)}{r} \leq \frac{M}{r}$$

    Hence $\partial f(a)$ is bounded.

#### Minimizer
- $x$ is a global minimizer of $f$ iff $0 \in \partial f(x)$.

#### Directional derivatives
- The following are equivalent:
  - $g \in \partial f(a)$
  - $g^T v \leq Df_a^+(v)$ for all $v \in \mathbb{R}^n$.
  - $Df_a^-(v) \leq g^T v \leq Df_a^+(v)$ for all $v \in \mathbb{R}^n$.
- For all $v \in \mathbb{R}^n$,
  $$Df_a^+(v) = \max_{g \in \partial f(a)} g^T v \text{  and  } Df_a^-(v) = \min_{g \in \partial f(a)} g^T v$$
  - **Proof** Suffices to prove it for $Df_a^+(v)$, since $Df_a^-(v) = -Df_a^+(-v)$. By last bullet point, we have 
    
    $$Df_a^+(v) \ge \max_{g \in \partial f(a)} g^T v$$
    
    and it suffices to show that equality holds. 

    Now for fixed $v$, define a linear map $l: \mathbb{R}v \to \mathbb{R}$ by $v \to Df_a^+(v)$. Clearly $l \leq Df_a^+$ on $\mathbb{R}v$ by construction. Since $Df_a^+$ is sublinear, by Hahn-Banach theorem this linear map $l$ can be extended to $\mathbb{R}$, which gives us precisely the subgradient extension we need.
- **Corollary** $f$ is differentiable at $a$ iff $\partial f(a)$ is a singleton.
  - **Proof of $\Leftarrow$**: It suffices to show that directional derivative in every direction exists. For the sake of contradiction, suppose directional derivative doesn't exist in direction $v$. Then $Df_a^+(v) \neq Df_a^-(v)$. But if $Df_a^+(v) = g^{+T} v$ and $Df_a^-(v) = g^{-T}(v)$ for $g^+, g^- \in \partial f(a)$, then $g^+ \neq g^-$, contradicting $\partial f(a)$ is singleton.

#### Creating more subdifferentiable functions

- Positive scaling: $\partial(\alpha f)(x) = \alpha \partial f(x)$ for $\alpha > 0$.
- Sum/integral: Let $f = \sum f_i$. If $g_i \in \partial f_i(x)$, then $\sum g_i \in \partial f(x)$.
- Affine linear Change of variable: Let $h(x) = f(Ax+b)$, where $A$ is a matrix and $b$ is a vector. If $g \in \partial f(Ax+b)$, then $A^T g \in \partial h(x)$.
  - We care about affine linear change because $h$ stays convex that way.
- Pointwise maximum over a finite set
  - **Theorem** Let $f = \max\\{f_1, \cdots, f_n\\}$. Then 
  
    $$\partial f(x) = co\left(\cup \left(\partial f_i(x) : f_i(x) = f(x)\right)\right)$$

    Here $co(A)$ means the convex hull of $A$. In other words, for a point $x$, if $f_i$'s are the "active" functions that realize $f(x) = \max\{f_1(x), \cdots, f_n(x)\}$, then $\partial f(x)$ is the convex hull of these $\partial f_i(x)$'s.

  - **Proof** 
    
    - Equality is clear if $S \subset \mathbb{R}$, so we will assume that case.
    - LHS contains RHS: It's clear that $\partial f(x)$ contains each $\partial f_i(x)$. Since $\partial f(x)$ is convex, it must contain the convex hull generated by these $\partial f_i(x)$.
    - RHS contains LHS: We will show that if $g \notin co\left(\cup \left(\partial f_i(x) : f_i(x) = f(x)\right)\right)$, then $g \notin \partial f(x)$.
   
      By hyperplane separation theorem (e.g. [Separation Theorem 1 in wikipedia page](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem)), we can find $v$ such that $g^T v < 0$, but $h^T v \ge 0$ for all $h \in co\left(\cup \left(\partial f_i(x) : f_i(x) = f(x)\right)\right)$. In particular, $h^T v \ge 0$ for all $h \in \partial f_i(x)$ where $f_i$ is active at $x$. Since we already know the statement for one-dimensional domain, we know the statement in the direction $v$, which would imply that 

      $$Df_x^- (v) = \min_{f_i \, active} D(f_i)_x^- (v) = \min_{f_i \, active} \min_{h \in \partial f(x)} h^T v \ge 0$$

      So $g^T v < Df_x^- (v)$, meaning that $g$ fails to be a subgradient in direction $v$. 
- Pointwise maximum over an infinite set
  - **Theorem (Ioff-Tihemirov)** Let $f_t: S \to \mathbb{R}$ be a family of convex functions, indexed by $t \in T$. Suppose that $T$ is a compact Huasdorff space, such that the map
 
    $$(t, x) \to f_t(x)$$

    is upper semicontinuous in $t$ for each $x$. Then

    $$\partial f(x) = clos\left(co\left(\cup \left(\partial f_t(x) : f_t(x) = f(x)\right)\right)\right)$$

    Note that we need an extra closure this time, because infinite union of closed sets may not be closed.

## Subgradients: examples

- Absolute value: The function $f(x) = \|x\|$ is not differentiable at 0, but is sub-differentiable at 0 with subgradient $\partial f(0) = [-1, 1]$. 
- $L^1$-norm: the function 
  
  $$f(x_1, \cdots, x_n) = |x_1| + \cdots + |x_n|$$

  is subdifferentiable. One way to look at this: this is the maximum of the $2^n$ functions $c_1x_1 + \cdots + c_nx_n$, where $c_i = \pm 1$; each of them is linear, hence convex differentiable, so the maximum is subdifferentiable as well. 

  What is the subdifferential at $(x_1, \cdots, x_n)$? 

  - Assuming that all $x_i \neq 0$, then the only active function is $sgn(x_1)x_1 + \cdots + sgn(x_n)x_n$, so the subgradient is $(sgn(x_1), \cdots, sgn(x_n))$.
  - In general if some $x_i = 0$, then both $+x_i$ and $-x_i$ can contribute, so for the $i$-th coordinate the contribution is then $[-1, 1]$ rather than $sgn(x_i)$.

## Subgradient descent

Let's first look at a typical gradient descent set up. Let $f: \mathbb{R}^n$ be convex differentiable. To run gradient descent, we initialize at a point $x^0$. At each step, we consider 

$$x^{i+1} = x^i - \alpha \nabla f(x^i)$$

where $\alpha$ is the step size. We want to say that after many iterations $x^i$ would be close to the global minimum $x^*$ of $f$. A typical convergence theorem is as follows.

**Theorem (gradient descent)** Suppose $f: \mathbb{R}^n \to \mathbb{R}$ is convex differentiable, with its gradient $L$-Lipschitz, i.e.

$$\|\nabla f(x) - f(y) \| \leq L \|x - y\|$$

for any $x,y$. If we run gradient descent with step size $\alpha \leq \frac{1}{L}$, then in $k$ steps, we have

$$f(x^k) - f(x^*) \leq \frac{\|x^0 - x^*\|^2}{2\alpha k}$$

**Proof** By Taylor expansion and $\nabla f$ being $L$-Lipschitz, we have 

$$f(y) \leq f(x) + \nabla f(x) \cdot (y-x) + \frac{1}{2} L \|y - x \|^2$$

This implies that

$$
\begin{align*}
f(x^{i+1}) - f(x^i)
&\leq \nabla f(x^i) \cdot (x^{i+1} - x^i) + \frac{1}{2} L\alpha^2 \|x^{i+1} - x^i \|^2 \\
&\leq - \alpha \|\nabla f(x_i)\|^2 + \frac{1}{2} L\alpha^2 \|\nabla f(x_i) \|^2 \\
&\leq - \frac{1}{2} \alpha \|\nabla f(x_i) \|^2
\end{align*}
$$

Therefore in each step, $f(x^{i+1})$ decreases. Moreover, by Taylor expansion at $x^i$,

$$f(x^i) \leq f(x^*) - \nabla f(x^i) \cdot (x^* - x^i)$$

Substituting and simplifying, we get

$$f(x^{i+1}) - f(x^*) \leq \frac{1}{2\alpha} \left(\|x^* - x^i\|^2 - \|x^* - x^{i+1}\|^2\right)$$

Sum it up for $i = 0, \cdots, k-1$, we get

$$(f(x^1) - f(x^*)) + \cdots (f(x^k) - f(x^*)) \leq \frac{1}{2\alpha} \left(\|x^* - x^0\|^2 - \|x^* - x^k\|^2\right) \leq \frac{1}{2\alpha} \|x^* - x^0\|^2$$

For LHS, since $f(x^i) - f(x^*)$ decreases as $i$ increases, we have

$$k (f(x^k) - f(x^*)) \leq \frac{1}{2\alpha} \|x^* - x^0\|^2$$

proving what we want. **QED**

Let's scrutinize the proof a little bit. We have the identity

$$f(x^{k+1}) - f(x^*) = \underbrace{f(x^k) - f(x^*)}_{1} + \underbrace{f(x^{k+1} - f(x^k)}_{2}$$

We used a first-order inequality to bound $(1)$ from the above, and a second-order inequality to bound $(2)$ from the above. Nonetheless it turns out that for the sake of telescoping, only looking at the first portion suffices. The kind of first-order inequality we need is exactly what subdifferentiability buys us, and this is why subgradient descent has hope to work.

**Theorem (subgradient descent)** Suppose $f: \mathbb{R}^n \to \mathbb{R}$ is convex subdifferentiable. Consider the update rule: initialize at $x^0$, and at each step consider

$$x^{i+1} = x^i - \alpha g^i$$

where $g^i \in \partial f(x^i)$. Suppose that $\\|g^i\\|_2 \leq G$ for all $i$. Then

$$\min\{f(x^1), \cdots, f(x^k)\} - f(x^*) \leq \frac{\|x^1 - x^*\|^2 + G^2 \alpha^2 k}{2 \alpha k}$$

In other words, this time we

- cannot guarantee $f(x^i)$ would decrease, but we can still look at the minimum so far.
- cannot guarantee that $f(x^i)$ would converge to $f(x^*)$, but it is within a constant distance (since RHS converges to $\frac{G^2\alpha}{2}$ as $k \to \infty$ - in particular as we choose a small enough step size $\alpha$, this distance can be as small as we want.)

**Proof** Direct modification of the proof for gradient descent.

By appropriately decreasing step size at each step (rather than using constant step size), one can make $f(x^k)$ converge to the minimum $f(x^*)$.
