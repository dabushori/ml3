import numpy as np
import sys
from scipy.special import softmax



def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return sigmoid(x) * (1-sigmoid(x))


class NeuralNetwork:
    def __init__(self, dims: list[int], activation_func, activation_der, eta):
        self.dims = dims
        self.eta = eta
        self.nlayers = len(dims)
        self.nclasses = dims[-1]
        self.w_layers = dict()
        self.bias_layers = dict()
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.w_layers[i] = np.zeros((d2, d1))
            self.bias_layers[i] = np.zeros((d2,1))
        self.activation = activation_func
        self.activation_der = activation_der

    def loss(self, h_last):
        res = -np.log(softmax(h_last))
        return np.sum(res, axis=0)
    
    def forward_prop(self, x, y):
        x = np.array([x]).T
        fprop_cache = {'x': x, 'h_0': x}
        h_prev = x
        for i in self.w_layers.keys():
            w = self.w_layers[i]
            b = self.bias_layers[i]
            z_i = np.dot(w, h_prev) + b
            fprop_cache[f'z_{i}'] = z_i
            h_i = self.activation(z_i)
            fprop_cache[f'h_{i}'] = h_i
            h_prev = h_i
        y_vec = np.zeros((self.nclasses,1))
        y_vec[y] = 1
        fprop_cache['y'] = y_vec
        fprop_cache['yhat'] = h_prev
        fprop_cache['loss'] = self.loss(h_prev)
        return fprop_cache
    
    def back_prop(self, fprop_cache):
        new_w = {i : np.copy(w) for i,w in self.w_layers.items()}
        new_b = {i: np.copy(b) for i,b in self.bias_layers.items()}
        dW_i = (fprop_cache['yhat'] - fprop_cache['y'])
        for i in reversed(range(1, self.nlayers)):
            new_w[i] -= self.eta * np.dot(dW_i, fprop_cache[f'h_{i - 1}'].T)
            new_b[i] -= self.eta * dW_i
            if i > 1:
                dW_i = np.dot(dW_i.T, self.w_layers[i]).T * self.activation_der(fprop_cache[f'z_{i-1}'])
        self.w_layers = new_w
        self.bias_layers = new_b

    def train(self, trainx_data, trainy_data):
        print('train started')
        for x, y in zip(trainx_data, trainy_data):
            fprop_cache = self.forward_prop(x, y)
            self.back_prop(fprop_cache)
        print('train ended')

    def predict(self, x, vec_return = False):
        h_prev = np.array([x]).T
        for i in self.w_layers:
            h_prev = self.activation(np.dot(self.w_layers[i], h_prev) + self.bias_layers[i])
        res = softmax(h_prev)
        return res if vec_return else np.argmax(res)


# main function
def main():
    # for debug
    if len(sys.argv) < 4:
        trainx_path = 'train_x'
        trainy_path = 'train_y'
        testx_path = 'test_x'
    else:
        trainx_path, trainy_path, testx_path = sys.argv[1:4]
    trainx_data = np.loadtxt(trainx_path)
    trainy_data = np.loadtxt(trainy_path, dtype=int)
    testx_data = np.loadtxt(testx_path)
    
    # # real code
    # if len(sys.argv) < 4:
    #     print('Usage: python ex3.py <train_x_path> <train_y_path> <test_x_path>')
    #     exit(0)
    
    # trainx_path, trainy_path, testx_path = sys.argv[1:4]
    # trainx_data = np.loadtxt(trainx_path)
    # trainy_data = np.loadtxt(trainy_path, dtype=int)
    # testx_data = np.loadtxt(testx_path)
        
    output_fname = sys.argv[4] if len(sys.argv) > 4 else 'test_y'
        
    with open(output_fname, 'w') as output_log:
        net = NeuralNetwork([784, 256, 256, 10], sigmoid, sigmoid_der, 0.5)
        net.train(trainx_data, trainy_data)
        for x in testx_data:
            print(f'{net.predict(x)}', file=output_log)
    
    # ex2 main
    # with open(output_fname, 'w') as output_log:
    #     # Exercise constants
    #     nclasses = 3
    #     dim = 5
    #     # KNN params
    #     knn_k = 5
    #     knn_norm = min_max
    #     knn = KNN(knn_k, nclasses)
    #     # Perceptron params
    #     perceptron_eta = 0.8
    #     perceptron_epochs = 200
    #     perceptron_k_folds = 17
    #     perceptron_norm = z_score
    #     perceptron = Perceptron(nclasses, perceptron_eta, perceptron_epochs, dim)
    #     # SVM params
    #     svm_eta = 0.1
    #     svm_lmbda = 0.001
    #     svm_epochs = 200
    #     svm_k_folds = 10
    #     svm_norm = z_score
    #     svm = SVM(nclasses, svm_eta, svm_lmbda, svm_epochs, dim)
    #     # PA params
    #     pa_epochs = 200
    #     pa_norm = z_score
    #     pa_k_folds = 10
    #     pa = PA(nclasses, pa_epochs, dim)

    #     # train all the algorithms (or set the data in KNN's case)
    #     knn.set_data(trainx_data.copy(), trainy_data.copy(), knn_norm)
    #     perceptron.train(trainx_data.copy(), trainy_data.copy(), perceptron_k_folds, perceptron_norm)
    #     svm.train(trainx_data.copy(), trainy_data.copy(), svm_k_folds, svm_norm)
    #     pa.train(trainx_data.copy(), trainy_data.copy(), pa_k_folds, pa_norm)

    #     if testx_data.ndim == 1:
    #         knn_yhat = knn.predict(testx_data)
    #         perceptron_yhat = perceptron.predict(testx_data)
    #         svm_yhat = svm.predict(testx_data)
    #         pa_yhat = pa.predict(testx_data)
    #         output_log.write(f'knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n')
    #     else:
    #         for x in testx_data:
    #             knn_yhat = knn.predict(x)
    #             perceptron_yhat = perceptron.predict(x)
    #             svm_yhat = svm.predict(x)
    #             pa_yhat = pa.predict(x)
    #             output_log.write(f'knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n')


if __name__ == '__main__':
    main()
