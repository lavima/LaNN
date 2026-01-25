from .modules import Module

class Sequence(Module):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def add(self, layer):
        self.layers.append(layer)

