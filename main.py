import gzip
import random
import math


class IDXDataset:
    def __init__(self, file_name: str):
        file: open = open(file_name, "rb")
        zipped_data: bytes = file.read()
        file.close()
        unzipped_data: bytes = gzip.decompress(data=zipped_data)
        self.dimensions: int = int.from_bytes(unzipped_data[3:4], byteorder="big")
        self.dimension_sizes: list[int] = [
            int.from_bytes(unzipped_data[4 + i : 8 + i], byteorder="big")
            for i in range(0, self.dimensions * 4, 4)
        ]
        self.data: list[int] = list(unzipped_data[self.dimensions * 4 + 4 :])


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
        self.biases: list[list[float]] = [
            [random.random() for _ in range(layer)] for layer in layers[1:]
        ]

    def squish_function(self, x: float) -> float:
        return 1 / (1 + math.exp(0.0 - x))

    def derivative_squish_function(self, x: float) -> float:
        return math.exp(0.0 - x) / ((1 + math.exp(0.0 - x)) ** 2)

    def feed_forward(self, input: list[float]) -> list[list[float]]:
        pass

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

    def train(self, training_images: IDXDataset, training_labels: IDXDataset):
        image_data: list[list[list[float]]] = self.prepare_training_images(
            training_images
        )
        label_data: list[list[float]] = self.prepare_training_labels(training_labels)
        num_batches: int = 500
        for batch in range(num_batches):
            pass

    def test(self, test_images: IDXDataset, test_labels: IDXDataset) -> float:
        image_data: list[list[list[float]]] = self.prepare_training_images(test_images)
        label_data: list[list[float]] = self.prepare_training_labels(test_labels)
        total_correct: int = 0
        total_wrong: int = 0
        for test in range(test_labels.dimensions[0]):
            pass
        return total_correct / (total_correct + total_wrong)


def main():
    training_images: IDXDataset = IDXDataset(file_name="train-images-idx3-ubyte.gz")
    print(training_images)
    training_labels: IDXDataset = IDXDataset(file_name="train-labels-idx1-ubyte.gz")
    print(training_labels)
    test_images: IDXDataset = IDXDataset(file_name="t10k-images-idx3-ubyte.gz")
    print(test_images)
    test_labels: IDXDataset = IDXDataset(file_name="t10k-labels-idx1-ubyte.gz")
    print(test_labels)
    network: Network = Network(layers=[784, 16, 16, 10])
    network.train(training_images=training_images, training_labels=training_labels)


if __name__ == "__main__":
    main()
