"""
Module for the Network class which represents a neural network
"""
import math
import random

from idx_dataset import IDXDataset
from matrix_operations import (
    add_matrix_matrix,
    apply_function_matrix,
    multiply_matrix_matrix,
)


class Network:
    """
    A neural network class which takes in layer configuration to create and train network
    """

    def __init__(self, layers: list[int]):
        self.layers: list[int] = layers
        self.weights: list[list[list[float]]] = [
            [[random.random() for _ in range(layers[i])] for _ in range(layers[i - 1])]
            for i in range(1, len(layers))
        ]
        self.biases: list[list[list[float]]] = [
            [[random.random()] for _ in range(layer)] for layer in layers[1:]
        ]

    def squish_function(self, x: float) -> float:
        return 1 / (1 + math.exp(0.0 - x))

    def derivative_squish_function(self, x: float) -> float:
        return math.exp(0.0 - x) / ((1 + math.exp(0.0 - x)) ** 2)

    def feed_forward(self, input: list[float]) -> list[list[float]]:
        activations: list[list[float]] = [input]
        for layer in range(len(self.layers) - 1):
            weights: list[list[float]] = self.weights[layer]
            biases: list[list[float]] = self.biases[layer]
            activations.append(
                apply_function_matrix(
                    add_matrix_matrix(
                        multiply_matrix_matrix(weights, activations[layer]), biases
                    ),
                    self.squish_function,
                )
            )

        print(activations)

    def prepare_training_images(self, training_images: IDXDataset) -> list[list[float]]:
        return [
            [
                float(
                    training_images.data[
                        i
                        * training_images.dimension_sizes[1]
                        * training_images.dimension_sizes[2]
                        + j * training_images.dimension_sizes[2]
                        + k
                    ]
                )
                / 255.0
                for k in range(training_images.dimension_sizes[2])
                for j in range(training_images.dimension_sizes[1])
            ]
            for i in range(training_images.dimension_sizes[0])
        ]

    def prepare_training_labels(self, training_labels: IDXDataset) -> list[list[float]]:
        return [
            [1.0 if j == training_labels.data[i] else 0.0 for j in range(10)]
            for i in range(training_labels.dimension_sizes[0])
        ]

    def train(self, training_images: IDXDataset, training_labels: IDXDataset):
        image_data: list[list[float]] = self.prepare_training_images(training_images)
        label_data: list[list[float]] = self.prepare_training_labels(training_labels)
        num_batches: int = 500
        for batch in range(num_batches):
            pass

    def test(self, test_images: IDXDataset, test_labels: IDXDataset) -> float:
        image_data: list[list[float]] = self.prepare_training_images(test_images)
        label_data: list[list[float]] = self.prepare_training_labels(test_labels)
        total_correct: int = 0
        total_wrong: int = 0
        for test in range(test_labels.dimensions[0]):
            pass
        return total_correct / (total_correct + total_wrong)
