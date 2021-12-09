# Ori Dabush 212945760
import numpy as np
import sys


def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return sigmoid(x) * (1-sigmoid(x))


def softmax(x):
    x = x - max(x)
    return np.exp(x)/sum(np.exp(x))


def shuffle(x, y):
    rand_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rand_state)
    np.random.shuffle(y)


class NeuralNetwork:
    def __init__(self, dims, activation_func, activation_der, eta):
        self.dims = dims
        self.eta = eta
        self.nlayers = len(dims)
        self.nclasses = dims[-1]
        self.w_layers = dict()
        self.bias_layers = dict()
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.w_layers[i] = np.random.normal(0, 0.1, (d2, d1))
            self.bias_layers[i] = np.random.normal(0, 0.1, (d2, 1))
        self.activation = activation_func
        self.activation_der = activation_der

    def loss(self, y, yhat):
        return float(-np.log(yhat[y]))

    def forward_prop(self, x, y, w_layers, bias_layers):
        x = np.array([x]).T
        fprop_cache = {'x': x, 'h_0': x}
        h_prev = x
        for i in w_layers.keys():
            w = w_layers[i]
            b = bias_layers[i]
            z_i = np.dot(w, h_prev) + b
            fprop_cache[f'z_{i}'] = z_i
            h_i = self.activation(z_i) if i != self.nclasses else softmax(z_i)
            fprop_cache[f'h_{i}'] = h_i
            h_prev = h_i
        y_vec = np.zeros((self.nclasses, 1))
        y_vec[y] = 1
        fprop_cache['y'] = y_vec
        fprop_cache['yhat'] = h_prev
        fprop_cache['loss'] = self.loss(y, h_prev)
        return fprop_cache

    def back_prop(self, fprop_cache, w_layers):
        dW_i = (fprop_cache['yhat'] - fprop_cache['y'])
        dW = {}
        db = {}
        for i in reversed(range(1, self.nlayers)):
            dW[i] = np.dot(dW_i, fprop_cache[f'h_{i - 1}'].T)
            db[i] = dW_i
            if i > 1:
                dW_i = np.dot(
                    dW_i.T, w_layers[i]).T * self.activation_der(fprop_cache[f'z_{i-1}'])
        return dW, db

    def divide_to_batches(self, trainx_data, trainy_data, batch_size):
        return [(trainx_data[i: i+batch_size], trainy_data[i: i+batch_size]) for i in range(0, len(trainx_data), batch_size)]

    def train(self, trainx_data, trainy_data, batch_size, epochs):
        for _ in range(epochs):
            shuffle(trainx_data, trainy_data)
            batches = self.divide_to_batches(
                trainx_data, trainy_data, batch_size)
            for batch_x, batch_y in batches:
                dW_total = {}
                db_total = {}
                for x, y in zip(batch_x, batch_y):
                    # forward propagation
                    fprop_cache = self.forward_prop(
                        x, y, self.w_layers, self.bias_layers)
                    # back propagation
                    dW, db = self.back_prop(fprop_cache, self.w_layers)
                    # sum the gradients (to average them)
                    for i in dW:
                        if i not in dW_total:
                            dW_total[i] = dW[i]
                            db_total[i] = db[i]
                        else:
                            dW_total[i] += dW[i]
                            db_total[i] += db[i]
                # update w and bias
                for i in dW_total:
                    self.w_layers[i] -= (self.eta / batch_size) * dW_total[i]
                    self.bias_layers[i] -= (self.eta /
                                            batch_size) * db_total[i]

    def predict(self, x):
        h_prev = np.array([x]).T
        for i in self.w_layers:
            h_prev = self.activation(
                np.dot(self.w_layers[i], h_prev) + self.bias_layers[i])
        res = softmax(h_prev)
        return np.argmax(res)


# main function
def main():
    if len(sys.argv) < 4:
        print('Usage: python ex3.py <train_x_path> <train_y_path> <test_x_path>')
        exit(0)

    trainx_path, trainy_path, testx_path = sys.argv[1:4]
    trainx_data = np.loadtxt(trainx_path) / 255
    trainy_data = np.loadtxt(trainy_path, dtype=int)
    testx_data = np.loadtxt(testx_path) / 255

    output_fname = sys.argv[4] if len(sys.argv) > 4 else 'test_y'

    with open(output_fname, 'w') as output_log:
        net_dims = [784, 128, 10]
        net_learning_rate = 0.01
        net = NeuralNetwork(net_dims, sigmoid, sigmoid_der, net_learning_rate)
        net_batch_size = 8
        net_epochs = 50
        net.train(trainx_data, trainy_data, net_batch_size, net_epochs)
        for x in testx_data:
            print(f'{net.predict(x)}', file=output_log)


if __name__ == '__main__':
    main()
