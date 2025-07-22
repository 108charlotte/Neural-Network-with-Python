import random
import os
print(os.getcwd())

file_path = os.path.join(os.path.dirname(__file__), "entries", "day_1.txt")
with open(file_path, 'r', encoding='utf-8') as file: 
    print("File opened")
    text = file.read()
    print("File contents loaded")

stopwords = ["the", "is", "and", "in", "at", "a", "to", "of"]

def generate_context_target_pairs(text, window_size): 
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]

    words = [word for word in words if word not in stopwords]
    
    pairs = []
    for i, target_word in enumerate(words): 
        first_word_index = max(0, i - window_size)
        last_word_index = min(i + window_size + 1, len(words))
        context_words = words[first_word_index:i] + words[i + 1:last_word_index]
        if len(context_words) == window_size * 2: 
            pairs.append((context_words, target_word))
    return pairs

def word_tokenize(text): 
    tokens = []
    curr_word = ""
    for char in text: 
        if char in [" ", "?", "."]: 
            tokens.append(curr_word)
            curr_word = ""
        else:
            curr_word += char
    return tokens

context_target_pairs = generate_context_target_pairs(text, window_size=2)
for context_words, target_words in context_target_pairs[:3]: 
    print(f'Context Words: {context_words}, Target Word: {target_words}')

# now, the model needs to take each word and using those context words try to determine the target word
# this will be accomplished using a simple neural network, which I will code from scratch

class NeuralNetwork(): 
    # num_neurons is a list with each item in the list corresponding to the number of neurons in that layer
    def __init__(self, num_neurons): 
        self.layers = []
        for i in range(len(num_neurons) - 1): 
            layer = Layer(num_neurons=num_neurons[i], num_neurons_next_layer=num_neurons[i+1])
            self.layers.append(layer)
    
    def visualize(self): 
        to_vis = ""
        for i in range(len(self.layers)): 
            neurons = self.layers[i].get_neurons()
            weights = []
            to_vis += "Layer " + str(i+1) + ": "
            for j in range(len(neurons)): 
                neuron = neurons[j]
                val = neuron.get_neuron_value()
                weights.append(neuron.get_neuron_weights())
                to_vis += str(round(val, 2)) + "    "
            to_vis += "\nWeights: "
            for k in range(len(neurons)): 
                to_vis += str(round(weights[k][0], 2)) + "    "
            to_vis += "\n"

        print(to_vis)
    
    def run_network(self, inputs): 
        total = 0
        for i in range(len(self.layers[0].get_neurons())): 
            self.layers[0].get_neurons()[i].set_neuron(inputs[i])
        for j in range(len(self.layers)): 
            if j == 0: 
                continue
            k = 0
            value = 0
            for neuron in self.layers[j].get_neurons(): 
                l = 0
                for prev_neuron in self.layers[j - 1].get_neurons(): 
                    value += prev_neuron.get_neuron_value() * prev_neuron.get_neuron_weights()[k]
                    l += 1
                value = self.layers[j].get_bias()
                neuron.set_neuron(value)
                k += 1
        return self.layers[-1]


class Layer(): 
    def __init__(self, num_neurons, num_neurons_next_layer): 
        self.neurons = []
        self.bias = random.random() * 2 - 1
        for i in range(num_neurons): 
            neuron = Neuron(num_outputs=num_neurons_next_layer)
            self.neurons.append(neuron)
    
    def get_neurons(self): 
        return self.neurons

    def get_bias(self): 
        return self.bias

class Neuron(): 
    def __init__(self, num_outputs): 
        self.weights = []
        self.outputs = num_outputs
        self.value = random.random() * 2 - 1
        for i in range(num_outputs): 
            self.weights.append((random.random() * 2) - 1)

    def set_neuron(self, value): 
        self.value = value
    
    def get_neuron_value(self): 
        return self.value
    
    def get_neuron_weights(self): 
        return self.weights

# define network
net = NeuralNetwork(num_neurons=[4, 1, 1])
net.visualize()
print(net.run_network([1, 1, 0, 1]))
net.visualize()