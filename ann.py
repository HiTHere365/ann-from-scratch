"""
ann.py

Simple 2-layer Artificial Neural Network (1 hidden layer) using NumPy.
Trains with static backpropagation to predict the next number in a short
increasing sequence (e.g., [1, 2, 3] -> 4).

Features demonstrated:
- Input layer that accepts data as a matrix
- One hidden layer with an activation function
- Output layer
- Weights and biases between layers
- Feedforward: matrix multiplication + activation
- Loss computation (Mean Squared Error)
- Backpropagation: gradient descent weight/bias updates
- Training loop with >= 1000 iterations
"""

import numpy as np

# ----- Activation and loss functions -----


def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid with respect to its input.
    Here x is assumed to be sigmoid(z), so derivative is x * (1 - x).
    """
    return x * (1.0 - x)


def mse_loss(y_true, y_pred):
    """Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)


# ----- Neural network class -----


class SimpleTwoLayerANN:
    """
    Two-layer ANN (1 hidden layer) using NumPy.

    Input -> Hidden (sigmoid) -> Output (sigmoid)

    Shapes:
      X: (n_samples, input_size)
      W1: (input_size, hidden_size)
      b1: (1, hidden_size)
      W2: (hidden_size, output_size)
      b2: (1, output_size)
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, seed=42):
        np.random.seed(seed)
        self.learning_rate = learning_rate

        # Initialize weights with small random values; biases start at zero
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def feedforward(self, X):
        """
        Feedforward function:
        1. Multiply input by weights and add bias for hidden layer.
        2. Apply activation function on hidden layer.
        3. Multiply hidden output by weights and add bias for output layer.
        4. Apply activation to get final prediction.
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1           # (n_samples, hidden_size)
        self.a1 = sigmoid(self.z1)                       # activation

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2     # (n_samples, output_size)
        self.a2 = sigmoid(self.z2)                       # predicted output (ŷ)
        return self.a2

    def backpropagate(self, X, y_true):
        """
        Static backpropagation using gradient descent.

        Steps:
        - Compute output error and gradient at output.
        - Propagate error back to hidden layer.
        - Compute gradients for all weights and biases.
        - Update parameters with learning rate.
        """
        # Forward pass (ensure we have current activations)
        y_pred = self.feedforward(X)

        # Output layer error and delta
        error_output = y_true - y_pred          # (n_samples, output_size)
        d_output = error_output * sigmoid_derivative(y_pred)

        # Hidden layer error and delta
        error_hidden = np.dot(d_output, self.W2.T)       # (n_samples, hidden_size)
        d_hidden = error_hidden * sigmoid_derivative(self.a1)

        # Gradients for weights and biases
        grad_W2 = np.dot(self.a1.T, d_output)            # (hidden_size, output_size)
        grad_b2 = np.sum(d_output, axis=0, keepdims=True)

        grad_W1 = np.dot(X.T, d_hidden)                  # (input_size, hidden_size)
        grad_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Gradient descent parameter update
        self.W2 += self.learning_rate * grad_W2
        self.b2 += self.learning_rate * grad_b2
        self.W1 += self.learning_rate * grad_W1
        self.b1 += self.learning_rate * grad_b1

        # Return current loss for monitoring
        return mse_loss(y_true, y_pred)

    def train(self, X, y, epochs=2000, verbose_every=200):
        """
        Train the network for a fixed number of epochs.

        Repeats feedforward + backprop at least 'epochs' times.
        """
        for epoch in range(epochs):
            loss = self.backpropagate(X, y)
            if verbose_every is not None and epoch % verbose_every == 0:
                print(f"Epoch {epoch:4d}  Loss: {loss:.6f}")

    def predict(self, X):
        """Return predicted outputs for given inputs X."""
        return self.feedforward(X)


# ----- Helper functions for sequence data -----


def build_training_data():
    """
    Build a simple training set where each input is a short increasing
    sequence and the target is the next number.

    Example:
      [1, 2, 3] -> 4
      [2, 3, 4] -> 5
      ...
    """
    sequences = []
    targets = []

    # Simple linear sequences from 1 to 10
    # windows of length 3 predicting the 4th value
    for start in range(1, 8):  # 1..7 gives sequences up to [7,8,9] -> 10
        seq = [start, start + 1, start + 2]
        next_val = start + 3
        sequences.append(seq)
        targets.append([next_val])

    X = np.array(sequences, dtype=float)
    y = np.array(targets, dtype=float)

    # Normalize roughly to [0,1] range to fit sigmoid output better
    X_norm = X / 10.0
    y_norm = y / 10.0
    return X_norm, y_norm


def get_user_input_sequence(input_size=3):
    """
    Ask the user for a comma-separated sequence and return as a NumPy array.
    Example input: "3,4,5"
    """
    raw = input(f"Enter {input_size} numbers separated by commas (e.g., 3,4,5): ")
    parts = raw.strip().split(",")
    if len(parts) != input_size:
        raise ValueError(f"Expected exactly {input_size} numbers.")

    seq = [float(p.strip()) for p in parts]
    arr = np.array(seq, dtype=float).reshape(1, -1)
    return arr


def main():
    # 1. Build training data
    X_train, y_train = build_training_data()
    n_samples, input_size = X_train.shape
    output_size = y_train.shape[1]
    hidden_size = 8  # small hidden layer

    print("Training data examples (normalized):")
    print("X_train[0]:", X_train[0], "-> y_train[0]:", y_train[0])

    # 2. Create and train the ANN
    ann = SimpleTwoLayerANN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=0.5,
        seed=0,
    )

    print("\nTraining the neural network...")
    ann.train(X_train, y_train, epochs=2000, verbose_every=400)

    # 3. Simple CLI for user input
    print("\nTraining complete.")
    print("Now you can enter a short increasing sequence, and the network")
    print("will predict the next number in the series.\n")

    try:
        user_seq = get_user_input_sequence(input_size=input_size)
    except ValueError as e:
        print("Input error:", e)
        return

    # Normalize user input in the same way as training data
    user_seq_norm = user_seq / 10.0

    pred_norm = ann.predict(user_seq_norm)
    # Denormalize prediction
    pred = pred_norm * 10.0

    print(f"\nInput sequence: {user_seq.flatten().tolist()}")
    print(f"Predicted next value: {float(pred.flatten()[0]):.3f}")


if __name__ == "__main__":
    main()