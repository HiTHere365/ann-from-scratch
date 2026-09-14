"""Tests for ann.py: shapes, training, the sigmoid derivative, and the seeded CLI example."""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ann import (  # noqa: E402
    SimpleTwoLayerANN,
    build_training_data,
    mse_loss,
    sigmoid,
    sigmoid_derivative,
)

# Hyperparameters used by main() in ann.py; the README sample output was produced with these.
HIDDEN_SIZE = 8
LEARNING_RATE = 0.5
SEED = 0
EPOCHS = 2000

# Value printed in the README "Sample Output" block for the input 5,6,7.
README_PREDICTION = 8.198


def _trained_network():
    X, y = build_training_data()
    ann = SimpleTwoLayerANN(
        input_size=X.shape[1],
        hidden_size=HIDDEN_SIZE,
        output_size=y.shape[1],
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    ann.train(X, y, epochs=EPOCHS, verbose_every=None)
    return ann


def test_forward_pass_output_shape():
    X, y = build_training_data()
    ann = SimpleTwoLayerANN(input_size=3, hidden_size=HIDDEN_SIZE, output_size=1, seed=SEED)
    out = ann.feedforward(X)
    assert X.shape == (7, 3)
    assert out.shape == (X.shape[0], 1) == y.shape
    assert np.all((out > 0) & (out < 1))


def test_loss_decreases_over_training():
    X, y = build_training_data()
    ann = SimpleTwoLayerANN(
        input_size=X.shape[1],
        hidden_size=HIDDEN_SIZE,
        output_size=y.shape[1],
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    loss_epoch_0 = ann.backpropagate(X, y)
    for _ in range(EPOCHS - 1):
        final_loss = ann.backpropagate(X, y)
    assert final_loss < loss_epoch_0
    assert final_loss < 0.1 * loss_epoch_0
    # The loss reported after training matches a fresh forward pass.
    assert np.isclose(mse_loss(y, ann.predict(X)), final_loss, rtol=1e-2)


def test_sigmoid_derivative_matches_finite_difference():
    z = np.linspace(-6.0, 6.0, 25)
    h = 1e-5
    numerical = (sigmoid(z + h) - sigmoid(z - h)) / (2.0 * h)
    # sigmoid_derivative expects the sigmoid output, not the pre-activation.
    analytical = sigmoid_derivative(sigmoid(z))
    np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-8)


def test_seeded_prediction_matches_readme_example():
    ann = _trained_network()
    pred = ann.predict(np.array([[5.0, 6.0, 7.0]]) / 10.0) * 10.0
    assert round(float(pred.flatten()[0]), 3) == README_PREDICTION


def test_cli_prints_readme_prediction():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "ann.py")],
        input="5,6,7\n",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    assert lines[-2] == "Input sequence: [5.0, 6.0, 7.0]"
    assert lines[-1] == f"Predicted next value: {README_PREDICTION:.3f}"
