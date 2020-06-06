---
layout: post
title: "Basic probability 1: notions"
page_name: Basic-probability-1--notions
date: 2020-06-02
categories: data-science basics probability
---

Main reference is [Amir Dembo's notes](http://web.stanford.edu/class/stats310a/lnotes.pdf). This post covers

- Probability space (assuming measure theory)
- Random variables
- Expectation and integration theory
- Independence
- Examples
- Convergence of random variables
- Conditional probability and conditional expectations

---


## Some toy examples

**Birthday Problem** 
How many people must be gathered together in a room, before you can be certain that there is a greater than 50/50 chance that at least two of them have the same birthday?

**Marriage Problem**
You meet $n$ people in your life. All these $n$ people can be ranked top to bottom in terms of compatibility with you as a soulmate, but you wouldn't know this exact rank until you meet all of them. These $n$ people come in your life in a random order, and you must accept/reject the person as a soulmate after meeting; you won't meet the person ever again after a rejection. What should be your strategy to find the best soulmate you can have?

**Monty Hall Problem**
Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?

## Probability space

Intuitively, probability is a number in $[0,1]$ that we assign to an event, with $0$ meaning an event can never happen, and $1$ meaning an event always happens [^1]. It should be additive in some sense, because if $A, B$ are "comparable" yet disjoint events, we would expect

$$\mathbb{P}(A \text{ or } B) = \mathbb{P}(A) + \mathbb{P}(B)$$

This naturally parallels the notion of length/area/volume, or more generally the notion of measure. 

**Example (Dice roll)**
Roll a fair dice. Intuitively, the probability of getting any of 1, 2, 3, 4, 5, 6 should be the "same" and is thus $\frac{1}{6}$. We should also be able to talk about probability of getting 1 or 6, probability of getting an even number, and so on.

At this point it's good to clarify what data is needed here, as well as potential issue that may occur.

**We need**
- **Sample space**: a set $\Omega$ of all possible outcomes. 
  - For dice roll, an outcome would be 1, 2, 3, 4, 5 or 6, so naturally the space of outcomes should be $\{1, 2, 3, 4, 5, 6\}$.
  - For the birthday problem, assume there are $n$ people in the room. Then an outcome would be a birthday assignment to each of the $n$ people.
- **Events**: a set $\mathcal{F}$ of events, with each event being a set of outcomes, i.e. a subset of $\Omega$. An event is what we can assign probability to. 
  - Note that we need to think about set of outcomes rather than outcome itself. 
    - In the dice roll example, we are interested in talking about probability of getting 1 or 6; but "1 or 6" is not an outcome of a dice roll, but rather than set of outcomes.
    - Similarly in the birthday problem, we are interested in talking about probability of at least two people having the same birthday. There are multiple birthday assignments that can do that.
  - Intuitively we should be able to consider and assign probability to any set of outcomes, i.e. $\mathcal{F} = 2^{\Omega}$. Unfortunately, measure theory suggests that this may not be possible sometimes even in nice cases, such as $\mathbb{R}$. Thus it is necessary to specify upfront for what we can talk about probability.
- **Probability**: a measure $\mu$ on $(\Omega, \mathcal{F})$ with total measure 1. Intuitively, $\mathbb{P}(\Omega)$ measures probability that some outcome occurs, which should always happen, given that you already pinned down the space of all outcomes to start with. Hence the total measure should be 1. Such measure is also called a probability measure.
  - This also effectively means probability space is, after rescaling, just a measure space with finite total measure.

**Definition (Probability space)** The datum $(\Omega, \mathcal{F}, \mu)$ is called a probability space. To conform to standard notations, we may use $\mathbb{P}$ as a synonym to $\mu$. 

**Remark** 
- To define a probability measure, we need to assign probability measure of all the events. This is often not feasible; generally $\mathcal{F}$ is specified by generators (as a $\sigma$-algebra), and it may be clear how to define a measure on the generators, but often it's unclear how to define it for the generated sets. The usual resolution is to consider a special set of generators, such as an algebra, and prove that measure defined on this special set of generator extends uniquely to the whole $\mathcal{F}$. One such theorem is [Caratheodory's extension theorem](https://en.wikipedia.org/wiki/Carath%C3%A9odory%27s_extension_theorem).
  - This is also where Dynkin's $\pi$-system and $\lambda$-system comes in.
 

## Random variables

Intuitively, random variable is a function on the outcomes.

**Example (Dice roll with payout)**
Roll 10 fair dices. We can ask questions like
- How many 6s are there on average?
  - This is a question about 
  
  $$\sum_{roll} 1_{\text{roll is 6}} = \begin{cases}1 & \text{roll is 6} \\ 0 & \text{otherwise}\end{cases}$$

- If rolling n gives me $n^2$ dollar, what is the average payout?
  - This is a question about 
  
  $$\sum_{roll} \sum_{n=1}^6 n^2 \cdot 1_{\text{roll is n}}$$

The example shows us that many natural questions we may formulate are questions about random variables. Naively, the very first thing we want to know about random variable is its average. The second thing is perhaps its variance. But in any case, averaging should be a central concept here. Given that we already have a probability space, measure theory says we are in a good spot to integrate functions here, but we have to restrict the class of integrable functions. Hence

**Definition (Random variable)** A random variable on $(\Omega, \mathcal{F}, \mu)$ is a measurable function from $X \to (\mathbb{R}, Lebesgue)$.

**Remark** We use $(\mathbb{R}, Lesbesgue)$ as the codomain so that we can do integration and take limits. (Strictly speaking, the codomain is really $[-\infty, \infty]$) For the basic theory of random variables, there are more general codomains we can use. (e.g. any measure space; for the sake of taking limits, any topological space with Borel $\sigma$-algebra, etc)

Ideally random variables should be a large class of functions, else the theory is useless. So the first question to ask is: what are some examples of random variables, and how do you cook up more of them?
- Simple functions 

  $$c_1 1_{A_1} + \cdots + c_k 1_{A_k}$$

  where $c_i \in \mathbb{R}$ and $A_i$ are events, are random variables.
- Composition: if $f: \mathbb{R} \to \mathbb{R}$ is continuous, $X$ is a random variable, then $f \circ X$ is also a random variable.
- Limits: if $X_1, X_2, \cdots$ is a sequence of random variables, then

$$\inf_i X_i, \sup_i X_i, \liminf_i X_i, \limsup_i X_i$$

are all random variables. Hence if $\lim_i X_i$ exists, it is also a random variable.

### Distribution, density and law

Let $X:(\Omega, \mathcal{F}, \mu)  \to (\mathbb{R}, Lebesgue)$ be a random variable. So far we don't need to consider the probability $\mu$ on $X$ at all. But if we mainly care about integration, then we care about $\mu$, since we need that to talk about 

$$\int_{\Omega} X \mu$$

But wait! We actually need something weaker: by change of variables formula, what we really care about is the pushforward measure $X_* \mu$, because:

$$\int_{\Omega} X \mu = \int_{\mathbb{R}} X_* \mu$$

**Definition (Law)** The law of a real-valued random variable $f:(\Omega, \mathcal{F}, \mu)  \to (\mathbb{R}, Lebesgue)$ is the pushforward measure $X_* \mu$ on $\mathbb{R}$.

Lebesgue measurable sets are nice: as long as we know the measure on all open intervals (or closed intervals), then the measure is pinned down. This gives us an alternative description of the law.

**Definition (Distribution)** The distribution function $F_X$ of a real-valued random variable $X$ is 

$$F_X(a) = X_*{\mu} ((-\infty, a)) = \mu (X^{-1} ((-\infty, a)) = \mathbb{P}(\omega \in \Omega: X(\omega) \leq a)$$

We can characterize all distribution functions directly:

**Theorem(Dembo, 1.2.37)** $F: \mathbb{R} \to [0,1]$ is a distribution function of some random variable, iff
- $F$ is non-decreasing
- $\lim_{x \to \infty} F(x) = 1$, $\lim_{x \to -\infty} F(x) = 0$
- $F$ is right continuous.

Finally, there is a special class of measures on $(\mathbb{R}, Lebesgue)$: those of the form $f(x) dx$, where $dx$ is the Lebesgue measure on $\mathbb{R}$. This is exactly the measures on $\mathbb{R}$ that are absolutely continuous wrt $dx$, by Radon-Nikodym theorem.

**Definition (Density)** The density $f_X$ of a real-valued random variable $X$, if exists, is a function such that $X_* \mu = f(x) dx$. In terms of the distribution, this means $f_X$ is a function such that 

$$F_X(a) = \int_{-\infty}^a f_X(x) dx$$

For $\int_{-\infty}^a f(x)dx$ to be a distribution, by the theorem we see that $f(x)$ must be Lebesgue measurable, non-negative a.e., with total integral 1.

### Connecting to discrete random variable

### Connecting to continuous random variable

# Footnotes
[^1]: Yes this is wrong, but at least this is my intuition when I was 5.
