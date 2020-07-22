---
layout: post
title: "CIFAR-10 #3a - Neural nets: notes"
page_name: CIFAR-10--3a---Neural-nets--notes
date: 2020-07-08
categories: data-science cs231n
---

## Gradients

### Backbone: gradient computation

- Backprop = chain rule. If the derivative with respect to each variable only depends on "forward-looking" values or constants, then by chain rule we can compute the derivative in backward direction.
- For neural nets, where you go from one layer to another via activation composed with linear combination of parameters, "hidden" in the neural net picture but is what's being used, is the derivatives with respect to all the weights involved in each layer (so that gradient descent can be done)

### Gradient-descent methods

- Gradient descent updates:
  - Update gradient after each data point
  - Update gradient after a small set of data point (mini-batch gradient descent)
    - Intuitively, this should be less accurate, and should be thought of as a tradeoff between speed and accuracy. 
      - When batch size is small, assuming continuity of derivatives you can argue that function values (without gradient update) is close enough to function values (with gradient update). So updating gradient after a batch doesn't matter.
      - This claim gets weaker as batch size increases. (e.g. I have no intuition why batch gradient descent makes any sense)
  - Update gradient after running through all data points once (batch gradient descent)
- Gradient descent issues
  - Vanishing gradient problem (e.g. with sigmoid activation)
  - Exploding gradient problem (e.g. RNN)

## Architecture

### Activation

- Sigmoid
  - Vanishing gradient problem
    - Output of a node is passed in to next layer, generally used in some linear combination. This is why people think of output $0$ as "inactivated", and output away from 0 as "activated".
    - When sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ is used as activation function, since 
      
      $$\sigma'(x) = \sigma(x) (1 - \sigma(x))$$

      So when $\sigma(x) \sim 0$, so is $\sigma'(x)$, and so is the gradient for weight/bias leading to this node.

      Now $\sigma(x) \sim 0$ must mean the linear combination $\sum_i w_i x_i + b \ll 0$ (here $x_i$ input from previous layer, $w_i$ weights, $b$ bias). Assuming that distribution of $x_i$ stays similar, and that this is caused by $b$ being very negative - then the situation won't change after a gradient update, since the gradient for weight/bias are both small. In other words, we are stuck in this node outputting a value $\sim 0$ forever (or for a long time), making it "inactivated". 
    - Naive solution is to decouple $h(x) \sim 0$ and $h'(x) \sim 0$ if $h(x)$ is the activation function. Common way to do this is to use $\tanh(x) = 2 \sigma(2x) - 1$, where if $\tanh(x) \sim 0$, then $\tanh'(x)$ is actually maximized.
  
- ReLU
  - Does sparsity play a role?
    - [Bengio's original paper](https://msngr.com/-GPG2NUAb5HpwFQK?funnel_session_id=_1d15fd3e-a2a0-49f6-bfc2-8df09424d77e) says yes. Summary at [stackexchange](https://stats.stackexchange.com/a/176905/137568).
    - [Ian Goodfellow thinks no however](https://www.quora.com/Why-does-ReLU-work-with-backprops-if-its-non-differentiable/answer/Chomba-Bupe/comment/36212566), the argument being: empirically leaky ReLU has comparable/better performance than ReLU, but doesn't have sparsity. See e.g. https://arxiv.org/pdf/1505.00853.pdf as well.
  - Dying ReLU problem
    - Dying neuron:
      - Node outputs 0 no matter what input is, which happens when bias is too negative. This can happen if learning rate is too high.
      - Can be visualized: weights/biases leading to the neuron would no longer update.
    - Dying layer
      - Whole layer outputs 0 no matter what input is. This can easily happen when number of layers is much bigger than number of nodes in each layer (unlikely to happen in real situation though).
        - Example: train 10 layer neural net with 2 nodes per layer, to learn $y = \vert x \vert$. High probability to learn constant function.
          - https://arxiv.org/pdf/1903.06733.pdf
        - Slightly different from dying neuron: whole layer outputting 0 would happen by chance as number of layers $\to \infty$. 
        - Similar to dying neuron: when a 0-layer occurs, This would kill gradients in all layers up to the 0-layer, and so previous weights/biases cannot be updated. So if by chance, the bias for the 0-layer is too negative (and thus always output 0), this can't be fixed.
  - Naive fix is to give some gradient when $x < 0$, so that there's a chance for update to happen. Common solutions in this regard include 
    - Leaky ReLU, 
    - [ELU](https://arxiv.org/pdf/1511.07289.pdf), 
    - [SELU](https://arxiv.org/pdf/1706.02515.pdf),
    - [GELU](https://arxiv.org/pdf/1606.08415.pdf)
    - [Swish](https://arxiv.org/pdf/1710.05941.pdf)
      - Similar to sigmoid, would still have vanishing gradient problem.
    - softplus, etc.
    - See also general comparison papers:
      - https://arxiv.org/pdf/1804.02763.pdf
      - https://arxiv.org/pdf/1901.02671.pdf
  - See also [Karpathy's blogpost](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b).

### Regularization techniques
- Parameter norm penalization
  - Some methods
    - Global penalty of norm on weights/biases: L1/L2/elastic net, etc.
    - Max norm: clipping norm of weight at each node after update.
  - Regularize per layer
     - Scale of parameters in each layer could be different. If we use the same hyperparameter to control size of weights across all layers, then layers with smaller weights are less regularized.
    - Solution: have separate hyperparameters for weight size for different layers. This is how tensorflow does it [by default](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer). ("Regularization penalties are applied on a per-layer basis")
  - Should one regularize bias as well?
- Ensemble methods
  - Bagging: resample training data and train
  - Dropout/Dropconnect
    - Dropout paper: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
      - Idea: in a layer, with probability $p$ set node output to be 0. Rescale output (or more simply, weights and biases) by $\frac{1}{p}$ so that expected value for next layer's input is the same.
      - Can think of this as an ensemble: if we apply dropout to a layer of $n$ nodes, think of it as an ensemble of $2^n$ combinations of (use/not use node) in the layer.
      - Misc
        - Section 7 is a good example on analyzing performance of neural network
- Modifying training set 
  - Resampling training data
  - Equivariant transformation on training set. For example in image recognition problem, scale/rotate/recolor training input while preserving labels.
  - Add noise to input data.
- Early stopping

### Optimizer

### Weight initialization

### Learning rate
- BatchNorm

## Miscellaneous

### Hyperparameter tuning techniques
