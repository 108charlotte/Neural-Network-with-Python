import random

class Neuron(): 
    inputs = []
    outputs = []
    weights = []
    bias = 0

    def __init__(self, n): 
        inputs = [0] * n
        outputs = [0] * n
        weights = [0.0] * n
        for i in range(n): 
            weights[i] = random.randint(50, 200) / 100
        bias = random.randint(50, 200) / 100


class Layer(): 
    neurons = []

class Network(): 
    layers = []