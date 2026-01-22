from lann.datasets import load_mnist

dataset = load_mnist()
print(dataset[0].shape)
print(dataset[1].shape)
