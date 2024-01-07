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
            [[random.random() for _ in range(layers[i - 1])] for _ in range(layers[i])]
            for i in range(1, len(layers))
        ]
        self.biases: list[list[list[float]]] = [
            [[random.random()] for _ in range(layer)] for layer in layers[1:]
        ]

    def squish_function(self, x: float) -> float:
        return 1 / (1 + math.exp(0.0 - x))

    def derivative_squish_function(self, x: float) -> float:
        return math.exp(0.0 - x) / ((1 + math.exp(0.0 - x)) ** 2)

    def feed_forward(self, input: list[list[float]]) -> list[list[float]]:
        activations: list[list[float]] = [[[value] for row in input for value in row]]
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
        return activations

    def prepare_training_images(
        self, training_images: IDXDataset
    ) -> list[list[list[float]]]:
        return [
            [
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
                ]
                for j in range(training_images.dimension_sizes[1])
            ]
            for i in range(training_images.dimension_sizes[0])
        ]

    def prepare_training_labels(self, training_labels: IDXDataset) -> list[list[float]]:
        return [
            [1.0 if j == training_labels.data[i] else 0.0 for j in range(10)]
            for i in range(training_labels.dimension_sizes[0])
        ]

    def back_propagate(
        self, activations: list[list[float]], label: list[float]
    ) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        return ([], [])

    def train(self, training_images: IDXDataset, training_labels: IDXDataset):
        image_data: list[list[list[float]]] = self.prepare_training_images(
            training_images
        )
        label_data: list[list[float]] = self.prepare_training_labels(training_labels)
        num_batches: int = 500
        items_per_batch: int = len(image_data) // num_batches
        for batch in range(num_batches):
            gradients: list[
                tuple[list[list[list[float]]], list[list[list[float]]]]
            ] = []
            for index, (image, label) in enumerate(
                zip(
                    image_data[batch * items_per_batch : (batch + 1) * items_per_batch],
                    label_data[batch * items_per_batch : (batch + 1) * items_per_batch],
                )
            ):
                activations: list[list[float]] = self.feed_forward(image)
                gradients.append(self.back_propagate(activations, label))
            number_of_gradients: int = len(gradients)
            for i in range(len(gradients[0])):
                for j in range(len(gradients[0][i])):
                    for k in range(len(gradients[0][i][j])):
                        self.weights[i][j][k] += (
                            sum(
                                [
                                    gradients[0][i][j][k][l]
                                    for l in range(gradients[0][i][j][k])
                                ]
                            )
                            / number_of_gradients
                        )
            for i in range(len(gradients[1])):
                for j in range(len(gradients[1][i])):
                    for k in range(len(gradients[1][i][j])):
                        self.biases[i][j][k] += (
                            sum(
                                [
                                    gradients[1][i][j][k][l]
                                    for l in range(gradients[1][i][j][k])
                                ]
                            )
                            / number_of_gradients
                        )

    def test(self, test_images: IDXDataset, test_labels: IDXDataset) -> float:
        image_data: list[list[float]] = self.prepare_training_images(test_images)
        label_data: list[list[float]] = self.prepare_training_labels(test_labels)
        total_correct: int = 0
        total_wrong: int = 0
        for image, label in zip(image_data, label_data):
            activations: list[list[float]] = self.feed_forward(image)
            if label.index(max(label)) == activations[-1].index(max(activations[-1])):
                total_correct += 1
            else:
                total_wrong += 1
        print(f"Total correct: {total_correct}")
        print(f"Total wrong: {total_wrong}")
        return float(total_correct) / float(total_correct + total_wrong)
