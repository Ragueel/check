import numpy as np
import random
import mnist_converter

input_layer = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
test_input = np.array([[0,0], [1,1], [0,0], [1,0], [0,1]])

output_layer = np.array([[0], [0], [0], [1]])
test_output = np.array([[0], [1], [0], [0], [0]])

network_size = np.array([64*64, 15, 3])
num_of_layers = len(network_size)

network_biases = [np.random.randn(y, 1) for y in network_size[1:]]
network_weights = [np.random.randn(y,x) for x, y in zip(network_size[:-1], network_size[1:])]

def convert_data(input_layer, output_layer):
    training_inputs = [np.reshape(x, (2, 1)) for x in input_layer]
    training_result = [vectorize(j) for j in output_layer]
    training_data = zip(training_inputs, training_result)
    return training_data


def vectorize(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative(activations, y):
    return (activations - y)

def backpropagation(x,y):
    nabla_b = [np.zeros(b.shape) for b in network_biases]
    nabla_w = [np.zeros(w.shape) for w in network_weights]

    activation = x
    activations = [x]
    zs = []

    for b, w in zip(network_biases, network_weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    delta = cost_derivative(activations[-1], y) * \
                                sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in xrange(2,num_of_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(network_weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return nabla_b, nabla_w

def update_mini_batch(mini_batch, eta):
    global network_biases, network_weights
    nabla_b = [np.zeros(b.shape) for b in network_biases]
    nabla_w = [np.zeros(w.shape) for w in network_weights]

    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backpropagation(x,y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    network_weights = [w - (eta / len(mini_batch))
                       * nw for w, nw in zip(network_weights, nabla_w)]
    network_biases = [b - (eta / len(mini_batch))
                      * nb for b, nb in zip(network_biases, nabla_b)]

def feedforward(a):
    global network_biases, network_weights
    for b, w in zip(network_biases, network_weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

def evaluate(test_data):
    test_results = [((np.argmax(feedforward(x)), y)) for (x, y) in test_data]
    sum = 0
    for x,y in test_results:
        if y[x] == 1:
            sum = sum + 1
    return sum

def SGD(training_data, number_of_epochs, mini_batch_size, eta, test_data=None):
    if test_data:
        n_test = len(test_data)
    n = len(training_data)
    for epoch in xrange(number_of_epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0,n,mini_batch_size)]
        for mini_batche in mini_batches:
            update_mini_batch(mini_batche, eta)
        if test_data:
            print "Epoch {0}: {1} / {2} ".format(epoch, evaluate(test_data), n_test)


converter = mnist_converter.Converter()
training_data = converter.getMNIST()
test_data = converter.getTest()
print training_data

SGD(training_data, 600, 6, 3.5, test_data=test_data)
print "Layout weights: {0}, Biases {1}".format(network_weights, network_biases)
