# ANN From Scratch: 2-Layer Neural Network in NumPy

A two-layer artificial neural network implemented from scratch using only NumPy. No PyTorch, no TensorFlow, no Keras — just the math.

Trains via backpropagation with gradient descent to predict the next value in a numeric sequence.

## What It Demonstrates

- **Feedforward pass**: matrix multiplication through input, hidden, and output layers
- **Sigmoid activation**: applied at each layer with its derivative computed manually
- **MSE loss**: mean squared error computed between predicted and true output
- **Backpropagation**: chain rule applied layer by layer to compute gradients
- **Gradient descent**: weights and biases updated each epoch by learning rate times gradient

## Architecture

```
Input (3) -> Hidden (8, sigmoid) -> Output (1, sigmoid)
```

Weights and biases are initialized with small random values. Training normalizes inputs and targets to [0, 1] to fit within sigmoid's output range.

## Requirements

```bash
pip install -r requirements.txt
```

Requires Python 3.7+.

## Running

```bash
python ann.py
```

## Sample Output

```
Training the neural network...
Epoch    0  Loss: 0.249583
Epoch  400  Loss: 0.008471
Epoch  800  Loss: 0.003204
Epoch 1200  Loss: 0.001847
Epoch 1600  Loss: 0.001201
Training complete.

Enter 3 numbers separated by commas (e.g., 3,4,5): 5,6,7
Input sequence: [5.0, 6.0, 7.0]
Predicted next value: 7.981
```

## Extending

- Swap sigmoid for ReLU by replacing the activation functions
- Add a second hidden layer by extending the weight matrices and backprop pass
- Replace the sequence task with any regression problem by swapping `build_training_data()`
- Add momentum or Adam by modifying the parameter update step in `backpropagate()`

## License

GNU Affero General Public License v3.0 (AGPL v3)

For commercial licensing: volts-beret0t@icloud.com
