import numpy as np
import matplotlib.pyplot as plt
import time

'''
    Basic neural network using sigmoid activation to classify gender
        - Male : 0
        - Female : 1
'''


def sigmoid(x):
    # Activation function (Sigmoid function): 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    # Derivative of sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_pred, y_true):
    # Mean squared error loss
    # y_true and y_pred are numpy arrays of same length
    return ((y_true - y_pred) ** 2).mean()


def classify(x):
    # Classifies whether feed_forward function outputs a male or female based on weight and height of individual
    if x >= .5:
        return 'F - ' + str(x)
    return 'M - ' + str(x)


class NeuralNetwork:
    """
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias
    """

    def __init__(self):
        # Weights and biases for hidden layer

        self.h_weights = np.matrix(
            [[np.random.normal(), np.random.normal()],
             [np.random.normal(), np.random.normal()]])  # 2 x 2
        # w1 w2
        # w3 w4

        self.h_biases = np.matrix(
            [[np.random.normal()],
             [np.random.normal()]])  # 2 x 1
        # b1
        # b2

        # Weights and biases for resulting layer
        self.o_weights = np.matrix(
            [np.random.normal(), np.random.normal()])  # 1 x 2
        # w5 w6

        self.o_bias = np.matrix(
            [np.random.normal()])  # 1 x 1
        # b3

    def feed_forward(self, x):
        '''
        Processes a weight and height and outputs whether the individual is male or female based on weights and biases
        of neural network
        :param x: weight and height to be processed (np.matrix size 2 x 1)
        :return: classification represented by a number between 0 and 1
        '''

        h_sum = np.matmul(self.h_weights, x) + self.h_biases  # 2 x 1
        h = sigmoid(h_sum)

        o_sum = np.matmul(self.o_weights, h) + self.o_bias  # 1 x 1
        o = sigmoid(o_sum)

        return o.item(0)

    def train(self, data, y_trues):
        """
        :param data: n x 2 numpy array, n = # of samples in dataset
        :param y_trues: numpy array with n elements
        """

        learn_rate = .1  # learning rate related to gradial descent function
        epochs = 1000  # number of times to loop through entire dataset

        x_axis = []
        y_axis = []

        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                h_sum = np.matmul(self.h_weights, x) + self.h_biases  # 2 x 1
                h = sigmoid(h_sum)

                o_sum = np.matmul(self.o_weights, h) + self.o_bias  # 1 x 1
                o = sigmoid(o_sum)

                y_pred = o.item(0)

                dL_ypred = -2 * (y_true - y_pred)

                # Neuron o

                # dL/dw = dL/d_ypred * J where J = grad(o) = [do/dw5, do/dw6, do/db]
                dypred_w = np.matrix(
                    [h.item(0) * sigmoid_der(o_sum.item(0)), h.item(1) * sigmoid_der(o_sum.item(0)),
                     sigmoid_der(o_sum.item(0))])
                dL_dw_o = dL_ypred * dypred_w

                # Neuron h

                dypred_h = np.matrix([self.o_weights.item(0) * sigmoid_der(o_sum.item(0)),
                                      self.o_weights.item(1) * sigmoid_der(o_sum.item(0))])
                dh_dw = np.matrix(
                    [[self.h_weights.item(0) * sigmoid_der(h_sum.item(0)),
                      self.h_weights.item(1) * sigmoid_der(h_sum.item(0)), sigmoid_der(h_sum.item(0))],
                     [self.h_weights.item(2) * sigmoid_der(h_sum.item(1)),
                      self.h_weights.item(3) * sigmoid_der(h_sum.item(1)), sigmoid_der(h_sum.item(1))]])

                dypred_h_v = np.matrix([[dypred_h.item(0) * dh_dw.item(0), dypred_h.item(0) * dh_dw.item(1),
                                         dypred_h.item(0) * dh_dw.item(2)],
                                        [dypred_h.item(1) * dh_dw.item(3), dypred_h.item(1) * dh_dw.item(4),
                                         dypred_h.item(1) * dh_dw.item(5)]])

                dL_dw_h = dL_ypred * dypred_h_v

                # Adjust weights and biases using stochastic gradient descent

                weights_h = np.matrix([[dL_dw_h.item(0), dL_dw_h.item(1)], [dL_dw_h.item(2), dL_dw_h.item(3)]])
                biases_h = np.matrix([[dL_dw_h.item(4)],
                                      [dL_dw_h.item(5)]])

                self.h_weights = self.h_weights - learn_rate * weights_h
                self.h_biases = self.h_biases - learn_rate * biases_h

                weights_o = np.matrix([dL_dw_o.item(0), dL_dw_o.item(1)])
                biases_o = np.matrix([dL_dw_o.item(2)])

                self.o_weights = self.o_weights - learn_rate * weights_o
                self.o_bias = self.o_bias - learn_rate * biases_o

            if epoch % 10 == 0:
                y_preds = np.array([self.feed_forward(data[0]), self.feed_forward(data[1]), self.feed_forward(data[2]),
                                    self.feed_forward(data[3])])
                x_axis.append(epoch)
                loss = mse_loss(y_preds, y_trues)
                y_axis.append(loss)
                print("Epoch %d loss: %3f" % (epoch, loss))

        print("Time to train: %3f sec" % (time.time() - start_time))
        plt.plot(x_axis, y_axis)
        plt.title("Training Loss over Time")
        plt.show()


# Define dataset
data = np.array([
    np.matrix([[-2], [-1]]),  # Alice
    np.matrix([[25], [6]]),  # Bob
    np.matrix([[17], [4]]),  # Charlie
    np.matrix([[-15], [-6]])  # Diana
])
all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# Initialize neural network
network = NeuralNetwork()

# Testing initial cases
emily = np.matrix([[-7], [-3]])  # 128 pounds, 63 inches
frank = np.matrix([[20], [2]])  # 155 pounds, 68 inches
print("Emily: " + classify(network.feed_forward(emily)))
print("Frank: " + classify(network.feed_forward(frank)))

# Train network
start_time = time.time()
network.train(data, all_y_trues)

# Testing cases
print("Emily: " + classify(network.feed_forward(emily)))
print("Frank: " + classify(network.feed_forward(frank)))
