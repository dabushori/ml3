import numpy as np, sys


class NeuralNetwork:
    def __init__(self, dims: list[int], activation_func):
        self.dims = dims
        self.nclasses = dims[-1]
        self.w_layers = dict()
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:])):
            self.w_layers[i] = np.zeros((d2,d1))
        self.activation = activation_func
        
    def loss(self, x, y, w):
        None
        
    def train(self, trainx_data, trainy_data):
        None
        
    def predict(self, x):
        None
        
# main function
def main():
    if len(sys.argv) < 4:
        print('Usage: python ex3.py <train_x_path> <train_y_path> <test_x_path>')
        exit(0)

    trainx_path, trainy_path, testx_path = sys.argv[1:4]
    trainx_data = np.loadtxt(trainx_path)
    trainy_data = np.loadtxt(trainy_path, dtype=int) # load classes as int instead of floats
    testx_data = np.loadtxt(testx_path)
    output_fname = 'test_y'
    
    
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