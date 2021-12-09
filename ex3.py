from operator import mul
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
        # mult_val = (1 / float(np.array(dims).sum())) ** 0.5
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.w_layers[i] = np.random.normal(0, 0.1, (d2, d1))
            self.bias_layers[i] = np.random.normal(0, 0.1, (d2, 1))
            # self.w_layers[i] = np.random.rand(d2, d1) * 0.001
            # self.bias_layers[i] = np.random.rand(d2, 1) * 0.001
        self.activation = activation_func
        self.activation_der = activation_der

    def loss(self, y, yhat):
        # return 1 if y != np.argmax(yhat) else 0
        #
        # results with 0-1 loss:
        # test accuracy: 3829 / 5000 (76.58%)
        # train accuracy: 42069 / 55000 (76.4890909090909%)
        #
        # results with NLL:
        # test accuracy: 4032 / 5000 (80.64%)
        # train accuracy: 43964 / 55000 (79.93454545454546%)
        #
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
        # yhat = softmax(h_prev)
        # fprop_cache['yhat'] = yhat
        fprop_cache['yhat'] = h_prev
        fprop_cache['loss'] = self.loss(y, h_prev)
        return fprop_cache

    def back_prop(self, fprop_cache, w_layers):
        # new_w = {i: np.copy(w) for i, w in self.w_layers.items()}
        # new_b = {i: np.copy(b) for i, b in self.bias_layers.items()}
        dW_i = (fprop_cache['yhat'] - fprop_cache['y'])
        dW = {}
        db = {}
        for i in reversed(range(1, self.nlayers)):
            dW[i] = np.dot(dW_i, fprop_cache[f'h_{i - 1}'].T)
            db[i] = dW_i
            # new_w[i] -= self.eta * np.dot(dW_i, fprop_cache[f'h_{i - 1}'].T)
            # new_b[i] -= self.eta * dW_i
            if i > 1:
                dW_i = np.dot(
                    dW_i.T, w_layers[i]).T * self.activation_der(fprop_cache[f'z_{i-1}'])
        # self.w_layers = new_w
        # self.bias_layers = new_b
        return dW, db

    def divide_to_batches(self, trainx_data, trainy_data, batch_size):
        return [(trainx_data[i: i+batch_size], trainy_data[i: i+batch_size]) for i in range(0, len(trainx_data), batch_size)]

    def train(self, trainx_data, trainy_data, batch_size, epochs):

        for j in range(epochs):
            print(f'epoch {j} started')
            shuffle(trainx_data, trainy_data)

            batches = self.divide_to_batches(
                trainx_data, trainy_data, batch_size)
            for batch_x, batch_y in batches:
                dW_total = {}
                db_total = {}
                for x, y in zip(batch_x, batch_y):
                    fprop_cache = self.forward_prop(
                        x, y, self.w_layers, self.bias_layers)
                    dW, db = self.back_prop(fprop_cache, self.w_layers)
                    for i in dW:
                        if i not in dW_total:
                            dW_total[i] = dW[i]
                            db_total[i] = db[i]
                        else:
                            dW_total[i] += dW[i]
                            db_total[i] += db[i]
                for i in dW_total:
                    self.w_layers[i] -= (self.eta / batch_size) * dW_total[i]
                    self.bias_layers[i] -= (self.eta /
                                            batch_size) * db_total[i]
            print(f'epoch {j} ended')

        # # for debug
        # for i in self.w_layers:
        #     w = self.w_layers[i]
        #     b = self.bias_layers[i]
        #     np.savetxt(f'w_{i}.txt', w)
        #     np.savetxt(f'b_{i}.txt', b)

    def predict(self, x, vec_return=False):
        h_prev = np.array([x]).T
        for i in self.w_layers:
            h_prev = self.activation(
                np.dot(self.w_layers[i], h_prev) + self.bias_layers[i])
        res = softmax(h_prev)
        return res if vec_return else np.argmax(res)


# main function
def main():
    # for debug
    start = time.time()
    if len(sys.argv) < 4:
        trainx_path = 'train_x'
        trainy_path = 'train_y'
        testx_path = 'test_x'
    else:
        trainx_path, trainy_path, testx_path = sys.argv[1:4]
    trainx_data = np.loadtxt(trainx_path) / 255
    trainy_data = np.loadtxt(trainy_path, dtype=int)
    testx_data = np.loadtxt(testx_path) / 255

    # # real code
    # if len(sys.argv) < 4:
    #     print('Usage: python ex3.py <train_x_path> <train_y_path> <test_x_path>')
    #     exit(0)

    # trainx_path, trainy_path, testx_path = sys.argv[1:4]
    # trainx_data = np.loadtxt(trainx_path) / 255
    # trainy_data = np.loadtxt(trainy_path, dtype=int)
    # testx_data = np.loadtxt(testx_path) / 255

    output_fname = sys.argv[4] if len(sys.argv) > 4 else 'test_y'

    # for debug (testing and shit)
    labels = np.loadtxt('test_labels')
    labels_hat = np.zeros(labels.shape)
    with open(output_fname, 'w') as output_log:
        net_learning_rate = 0.01
        net_batch_size = 8
        net_epochs = 50
        net_dims = [784, 128, 10]
        net = NeuralNetwork(net_dims, sigmoid, sigmoid_der, net_learning_rate)
        net.train(trainx_data, trainy_data, net_batch_size, net_epochs)
        for i, x in enumerate(testx_data):
            yhat = net.predict(x)
            labels_hat[i] = yhat
            print(f'{yhat}', file=output_log)
        delta = time.time() - start
        print(f'time: {delta} seconds, which are {delta // 60} minutes and {delta % 60} seconds')
        print()
        print(
            f'test accuracy: {(labels_hat == labels).sum()} / {labels_hat.shape[0]} ({np.round(100 * (labels_hat == labels).sum() / labels_hat.shape[0], 4)}%)')
        train_yhat = np.array([net.predict(x) for x in trainx_data])
        print(
            f'train accuracy: {(train_yhat == trainy_data).sum()} / {train_yhat.shape[0]} ({np.round(100 * (train_yhat == trainy_data).sum() / train_yhat.shape[0], 4)}%)')

    # # real code
    # with open(output_fname, 'w') as output_log:
    #     net = NeuralNetwork([784, 20, 10], sigmoid, sigmoid_der, 0.1)
    #     net_batch_size = 20
    #     net_epochs = 5
    #     net.train(trainx_data, trainy_data, net_batch_size, net_epochs)
    #     for x in testx_data:
    #         print(f'{net.predict(x)}', file=output_log)


# # for debug
# np.seterr(all='raise')
import time

if __name__ == '__main__':
    main()
