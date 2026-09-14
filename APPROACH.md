# Design Approach

## Why From Scratch

Most neural network code calls `model.fit()` and treats the training loop as a black box. Implementing the network from scratch in NumPy forces every component to be explicit: the weight matrices, the activation function, the loss computation, and each step of the gradient calculation. There is nowhere to hide if the math is wrong.

## Architecture

The network has two weight layers, so it is described as a two-layer network; counting the input, there are three layers of units:

```
X (input) -> z1 = X @ W1 + b1 -> a1 = sigmoid(z1) -> z2 = a1 @ W2 + b2 -> a2 = sigmoid(z2)
```

- **Input layer**: 3 neurons (three-step sequence window)
- **Hidden layer**: 8 neurons with sigmoid activation
- **Output layer**: 1 neuron with sigmoid activation (normalized target value)

Weights are drawn from a seeded normal distribution and scaled by 0.1; biases are initialized to zero.

Sigmoid was chosen because the targets are normalized to [0, 1], which fits within sigmoid's output range. For unbounded regression, a linear output layer would be more appropriate.

## Feedforward

Each forward pass computes two matrix multiplications:

```
z1 = X @ W1 + b1      # (n_samples, hidden_size)
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2     # (n_samples, output_size)
a2 = sigmoid(z2)      # prediction
```

## Loss

Mean Squared Error between predicted and true output:

```
L = mean((y_true - y_pred)^2)
```

MSE penalizes larger errors more heavily than smaller ones, which is appropriate for a regression task on a smooth numeric sequence.

## Backpropagation

Backprop applies the chain rule layer by layer to compute how much each weight contributed to the loss.

**Output layer:**
```
error_output = y_true - y_pred
d_output = error_output * sigmoid'(a2)
```

**Hidden layer:**
```
error_hidden = d_output @ W2.T
d_hidden = error_hidden * sigmoid'(a1)
```

**Gradients:**
```
grad_W2 = a1.T @ d_output
grad_b2 = sum(d_output)
grad_W1 = X.T @ d_hidden
grad_b1 = sum(d_hidden)
```

**Parameter update (gradient descent):**
```
W += learning_rate * grad_W
b += learning_rate * grad_b
```

The sign is positive here because the error is defined as `y_true - y_pred` rather than `y_pred - y_true`, so the quantity called `grad` is the negative of the loss gradient and adding it moves downhill. The constant factor `2 / n_samples` from differentiating the mean squared error is dropped; it is absorbed into the learning rate.

## Normalization

Inputs and targets are divided by 10 before training to bring values into the range where sigmoid operates effectively (avoiding saturation near 0 or 1, where gradients vanish). Predictions are multiplied back by 10 before display. Because the output is a sigmoid, the denormalized prediction is always strictly below 10; sequences beyond the training windows (`[1,2,3]` through `[7,8,9]`) are extrapolation and the network does not handle them well.

## Tradeoffs

- Sigmoid saturates for large inputs, causing vanishing gradients in deeper networks. ReLU is preferred in practice for hidden layers.
- Vanilla gradient descent updates weights once per pass over all seven training samples (full-batch gradient descent). Mini-batch or stochastic gradient descent converge faster on large datasets.
- Fixed learning rate can overshoot or converge slowly. Adaptive optimizers (Adam, RMSProp) adjust the rate per parameter.
- Training runs for a fixed 2000 epochs with no validation set or early stopping; the dataset is seven samples, so this is a teaching example rather than a benchmark.
