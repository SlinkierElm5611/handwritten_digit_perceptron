"""
Module to train and test a neural network on the MNIST dataset
"""
from idx_dataset import IDXDataset
from network import Network


def main():
    training_images: IDXDataset = IDXDataset(file_name="train-images-idx3-ubyte.gz")
    training_labels: IDXDataset = IDXDataset(file_name="train-labels-idx1-ubyte.gz")
    test_images: IDXDataset = IDXDataset(file_name="t10k-images-idx3-ubyte.gz")
    test_labels: IDXDataset = IDXDataset(file_name="t10k-labels-idx1-ubyte.gz")
    network: Network = Network(layers=[784, 16, 16, 10])
    network.train(training_images=training_images, training_labels=training_labels)
    print(network.test(test_images=test_images, test_labels=test_labels))


if __name__ == "__main__":
    main()
