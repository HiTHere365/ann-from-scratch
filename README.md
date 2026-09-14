# ANN From Scratch: 2-Layer Neural Network in NumPy

A two-layer artificial neural network implemented from scratch using only NumPy. No PyTorch, no TensorFlow, no Keras: just the math.

Trains via backpropagation with gradient descent to predict the next value in a short increasing integer sequence.

## What It Demonstrates

- **Feedforward pass**: matrix multiplication through input, hidden, and output layers
- **Sigmoid activation**: applied at the hidden and output layers, with its derivative computed manually
- **MSE loss**: mean squared error computed between predicted and true output
- **Backpropagation**: chain rule applied layer by layer to compute gradients
- **Gradient descent**: weights and biases updated each epoch by learning rate times gradient (full-batch, fixed learning rate)

## Architecture

```
Input (3) -> Hidden (8, sigmoid) -> Output (1, sigmoid)
```

"Two-layer" counts the two weight layers (input to hidden, hidden to output). Weights are initialized with small random values from a fixed seed; biases start at zero. Training divides inputs and targets by 10 so they fall inside sigmoid's (0, 1) output range.

## Requirements

```bash
pip install -r requirements.txt
```

The only runtime dependency is NumPy. Python 3.11 and 3.13 are exercised in CI; `pytest` is needed only for the test suite.

## Running

```bash
python ann.py
```

The script trains for 2000 epochs on the built-in data (the windows `[1,2,3] -> 4` through `[7,8,9] -> 10`), prints the loss every 400 epochs, then prompts for three comma-separated numbers and prints the predicted next value. To run it non-interactively, pipe the input:

```bash
printf '5,6,7\n' | python ann.py
```

## Sample Output

Exact output of `printf '5,6,7\n' | python ann.py`. Weight initialization uses a fixed seed, so repeated runs print identical numbers. When the input is piped, the typed values do not appear after the prompt.

```
Training data examples (normalized):
X_train[0]: [0.1 0.2 0.3] -> y_train[0]: [0.4]

Training the neural network...
Epoch    0  Loss: 0.061044
Epoch  400  Loss: 0.002767
Epoch  800  Loss: 0.001318
Epoch 1200  Loss: 0.001129
Epoch 1600  Loss: 0.001015

Training complete.
Now you can enter a short increasing sequence, and the network
will predict the next number in the series.

Enter 3 numbers separated by commas (e.g., 3,4,5): 
Input sequence: [5.0, 6.0, 7.0]
Predicted next value: 8.198
```

The true next value is 8; the network predicts 8.198. The training set only covers windows between `[1,2,3]` and `[7,8,9]`, and the sigmoid output layer cannot produce a value above 10, so inputs outside that range are extrapolation and will be inaccurate (for example, `20,21,22` predicts about 9.97).

## Tests

```bash
pip install pytest
pytest -q
```

The suite in `tests/test_ann.py` checks the forward-pass output shape, that the loss decreases over training on the built-in data, that the sigmoid derivative matches a numerical finite difference, and that the seeded prediction for `5,6,7` equals the 8.198 shown above (both in-process and through the CLI).

## Project Layout

```
ann.py                    network, training loop, and CLI
tests/test_ann.py         pytest suite
APPROACH.md               design notes
requirements.txt          runtime dependency (numpy)
.github/workflows/ci.yml  runs the tests on Python 3.11 and 3.13
```

## Extending

Ideas for follow-on work; none of these are implemented here:

- Swap sigmoid for ReLU by replacing the activation functions
- Add a second hidden layer by extending the weight matrices and backprop pass
- Replace the sequence task with any regression problem by swapping `build_training_data()`
- Add momentum or Adam by modifying the parameter update step in `backpropagate()`

## License

MIT. See [LICENSE](LICENSE).

For commercial licensing: volts-beret0t@icloud.com
