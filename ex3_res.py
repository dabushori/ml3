import sys
from numpy import genfromtxt
import numpy as np
from sklearn.utils import shuffle # not working on summit.biu.ac.il
from scipy import stats
import scipy.special as scipy
import pickle

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # x = normalize_values(x)
    return scipy.softmax(x)
    # maxvar = np.max(x)
    # e_x = np.exp(x - maxvar)
    # return e_x / e_x.sum(axis=0)


def derivative_softmax(A, y):
    mat = np.copy(A)
    mat[y]-=1
    return mat


def derivative_loss(pr_y):
    """
    not in use
    :param pr_y:
    :return:
    """
    # return y_hat - y
    if pr_y == 0 :
        pr_y=0.000000000000001
    loss = -(1/pr_y)
    return  loss

def ReLU(x) :
    return np.maximum(0,x)

def derivative_ReLU(x) :
    x= np.copy(x)
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def normalize_values(x):
    # return stats.zscore(x)
    # return x
    maxx =x.max()
    if maxx != 0:
        return x /maxx
    return x



class Network():
    """
    class that manage the network
    """
    random_w = 0
    noramlizaion_x = 255
    def __init__(self, neuron_len, data_len , learning_rate=0.0001, epochs=10, class_len = 10):
        """
        init the data
        :param neuron_len: how many neuwons in each hidden layer
        :param data_len: fow many feature in the data
        :param learning_rate: the learning rate
        :param epochs: epochs
        :param class_len: to how many class the data divide
        """
        np.random.seed(1)
        self.class_len=class_len
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.neuron_len=neuron_len
        self.loss = []

        ## for the next if i want to add more hidden layers
        self.network_dept = 1
        # self.bais = {}
        # self.weights = {}
        # for i in range(self.network_dept):   ## for the next if i want to add more hidden layers
        #     self.bais.update({i:np.random.uniform(-self.random_w, self.random_w, (neuron_len, 1))})
        #     self.weights.update({i:np.random.uniform(-self.random_w, self.random_w, (neuron_len, neuron_len))})

        self.bais_in = np.random.uniform(-self.random_w, self.random_w, (neuron_len, 1))
        self.weights_in = np.random.uniform(-self.random_w, self.random_w, (neuron_len, data_len))

        self.bais_out = np.random.uniform(-self.random_w, self.random_w, (self.class_len, 1))
        self.weights_out = np.random.uniform(-self.random_w, self.random_w, (self.class_len, neuron_len))

    def feedforward(self,v):
        """
        calculate the prediction
        :param v:
        :return:
        """
        self.v =v = np.array([v])
        self.z1 = np.dot(self.weights_in, v.T) + self.bais_in
        # self.z1 = normalize_values(self.z1)
        self.a1 = ReLU(self.z1)
        self.a1 = normalize_values(self.a1)

        # for the next if i want to add more hidden layers
        # self.z2 = np.dot(self.weights[0], self.a1) + self.bais[0]
        # self.a2 = ReLU(self.z2)

        self.z3 = np.dot(self.weights_out, self.a1) + self.bais_out
        # self.z3 = normalize_values(self.z3)
        self.a3 = softmax(self.z3)

        return self.a3

    def backpropagation1(self,x,y):
        """
        update the w
        :param x:
        :param y:
        :return:
        """
        # compute the gradient on scores
        # dloss = derivative_loss(self.a3[y]) #  dL/dytag
        # Gradient w.r.t parameters
        dl_yhat = derivative_softmax(self.a3,y)  #  dL/dytag
        # dl_yhat = self.a3 -y  #  dL/dytag
        db3 = dl_yhat
        dW3 = np.dot(dl_yhat, self.a1.T) #  dL/dytag

        #Gradient w.r.t input
        dl_input3 = np.dot(self.weights_out.T,dl_yhat)  #  dL/dytag

        ### for the next if i want to add more hidden layers
        # # Gradient w.r.t parameters
        # dh2_relu = dl_input3 * derivative_ReLU(self.z2) #  dL/dytag
        # dW2 = np.dot(dh2_relu,self.a1.T) #  dL/dytag
        # db2 = dh2_relu
        # # #Gradient w.r.t input
        # dh2 = np.dot(self.weights[0].T,dh2_relu)

        # Gradient w.r.t parameters
        dh1_relu = dl_input3 * derivative_ReLU(self.z1)
        dW1 = np.dot(dh1_relu,  x)
        db1 = dh1_relu

        self.weights_out -= dW3*self.learning_rate
        self.bais_out -= db3*self.learning_rate

        ### for the next if i want to add more hidden layers
        # self.weights[0] += -self.learning_rate * dW2
        # self.bais[0] += -self.learning_rate * db2
        ##

        self.weights_in -= dW1*self.learning_rate
        self.bais_in -= db1*self.learning_rate

    def backpropagation(self, x, y):
        e1 = self.a3
        e1[y] = e1[y] - 1
        g_l = np.reshape(e1, (self.class_len, 1))
        delta_b2 = g_l
        w2_transpose = np.transpose(self.weights_out)
        v1_transpose = np.reshape(np.transpose(self.a1), (1, self.neuron_len))
        delta_w2 = np.dot(g_l, v1_transpose)
        delta_b1 = np.dot(w2_transpose, e1)
        delta_b1 *= derivative_ReLU(self.z1)
        x_tag = np.reshape(x, (1, 28 * 28))
        delta_w1 = np.dot(np.reshape(delta_b1, (self.neuron_len, 1)), x_tag)

        self.weights_out += -self.learning_rate * delta_w2
        self.bais_out += -self.learning_rate * delta_b2
        self.weights_in += -self.learning_rate * delta_w1
        self.bais_in += -self.learning_rate * delta_b1

        return {'delta_w1': delta_w1, 'delta_w2': delta_w2, 'delta_b1': delta_b1, 'delta_b2': delta_b2}

    def predict(self,v):

        return np.argmax(self.feedforward(v))

    def train(self,x_train,y_train):
        """
        do the train
        :param x_train:
        :param y_train:
        :return:
        """
        i=0
        for _ in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train, random_state=1)
            total_loss = 0
            for x,y in zip(x_train,y_train):

                pr = self.feedforward(x)
                if pr[y] == 0:
                    # if the net got stuck
                    self.a3[y] += 0.000000000001
                    self.a3[np.argmax(pr)] -=0.000000000001

                loss = -np.log(pr[y])
                total_loss+=loss
                if loss != 0:
                    self.backpropagation(np.array([x]),y)
            i += 1
            total_loss /= x_train.shape[0]
            self.loss.append(float(total_loss))
            print (self.validate(self.x_vaild, self.y_vaild))
            print(f"epoch: {i} total loss:{total_loss/x_train.shape[0]}")

    def test(self, data,filename="test_y"):
        """
        write the test
        :param x_test:
        :param y_test:
        :return:
        """
        f = open(filename,"w")
        for x in data:
            y_hat= self.predict(x)
            f.write(f"{y_hat}\n")

    def validate(self, x_test, y_test):
        """
        only for debug
        :param x_test:
        :param y_test:
        :return:
        """
        self.num_examples = x_test.shape[0]
        right_cont=0
        # x_test, y_test = shuffle(x_test, y_test, random_state=1)
        for x,y in zip(x_test,y_test):
            y_hat= self.predict(x)
            if y ==y_hat:
                right_cont +=1

        print (f"(right in: {right_cont}) {round(right_cont /self.num_examples,2)*100}%")
        return right_cont /self.num_examples

    def train_and_validate(self, data_x, data_y, group_dived = 0.80):
        """
        only for debug
        :param data_x:
        :param data_y:
        :param group_dived:
        :return:
        """

        # data_x /= Network.noramlizaion_x
        num_examples = data_x.shape[0]
        train_size = round(num_examples * group_dived)
        self.x_vaild = x_vaild= data_x[train_size:num_examples]
        self.y_vaild = y_vaild  = data_y[train_size:num_examples]
        self.x_train = x_train = data_x[:train_size]
        self.y_train = y_train = data_y[:train_size]

        self.train(x_train, y_train)
        return self.validate(x_vaild, y_vaild)

    def print_loss_graph(self):
        """
        print some graphs for the documention
        :return:
        """
        import matplotlib.pyplot as plt
        plt.plot(range(self.epochs), self.loss)
        plt.ylabel('loss ')
        plt.xlabel('epoch ')
        # plt.xticks(plt_learning)
        plt.show()

def main():
    ## read the input data
    train_X = sys.argv[1]
    train_y = sys.argv[2]
    test_X = sys.argv[3]
    x_train = np.loadtxt(train_X)#genfromtxt(train_X, delimiter=' ')
    y_train = np.loadtxt(train_y, dtype=np.int32)#genfromtxt(train_y, delimiter=' ', dtype=np.int32)
    #normalize the input
    x_train /= Network.noramlizaion_x
    # x_train =stats.zscore(x_train,axis=1)
    #create the object
    hidden_size = [100]
    lr = [0.1]
    best_rate = 0
    best_model = None
    for h in hidden_size:
        for l in lr:
            print("----------------model hidden size {}, learning rate {}----------------".format(h, l))
            mynet = Network(h, x_train.shape[1], epochs=15,learning_rate=l)
            rate = mynet.train_and_validate(x_train, y_train)
            if rate > best_rate:
                print(f"found new params learnimg rate:{l} hidden size: {h} validation {rate}")
                best_rate = rate

                # mynet.print_loss_graph()
                # mynet.train(x_train, y_train)
                # x_test = genfromtxt(test_X, delimiter=' ')
                # x_test /= Network.noramlizaion_x
                # mynet.test(x_test)


if __name__ == "__main__":
    main()