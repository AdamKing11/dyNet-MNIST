import os, sys, re
import numpy
import csv
import pickle
from tqdm import tqdm
from pprint import pprint

from sklearn.metrics import log_loss

import dynet_config
dynet_config.set(
    mem=4096,          # can probably get away with 1024
    autobatch=True,    # utilize autobatching
    random_seed=1978   # simply for reproducibility here
)
import dynet


# 0 vs 1
def load(f = 'mnist_train.csv'):
    x_py = []
    y_py = []
    with open(f) as rf:
        reader = csv.reader(rf, delimiter = ',')
        for line in tqdm(reader):
            y_onehot = [0 for _ in range(10)]
            y_onehot[int(line[0])] = 1
            y_py.append(y_onehot)
            x_py.append(line[1:])
    return numpy.asarray(x_py, dtype = 'int8'), numpy.asarray(y_py, dtype = 'int8')

def load_mnist(rl = False, pick = False):
    print('training data....')
    if not rl:
        X = pickle.load(open('data/train_X.pick', 'rb'))
        y = pickle.load(open('data/train_y.pick', 'rb'))
    else:
        X,y = load('data/mnist_train.csv')
        if pick:
            pickle.dump(X, open('data/train_X.pick', 'wb'))
            pickle.dump(y, open('data/train_y.pick', 'wb'))
    print('testing data....')
    if not rl:
        test_X = pickle.load(open('data/test_X.pick', 'rb'))
        test_y = pickle.load(open('data/test_y.pick', 'rb'))
    else:
        test_X,test_y = load('data/mnist_test.csv')
        if pick:
            pickle.dump(test_X, open('data/test_X.pick', 'wb'))
            pickle.dump(test_y, open('data/test_y.pick', 'wb'))
    return (X,y), (test_X, test_y)


class cm:

    def __init__(self, input_size, hidden_size = 200, output_size = 10):
        self.model = dynet.ParameterCollection()   
        self.init = dynet.GlorotInitializer(gain=4.0)
        ####
        # add layers
        self.layers = []
        # first layer
        self.layers.append(self.add_layer(input_size, hidden_size))
        # output layer
        self.layers.append(self.add_layer(hidden_size, output_size))
        # 4.0 for logistic
        self.trainer = dynet.SimpleSGDTrainer(m=self.model, learning_rate=.001)


    def add_layer(self, in_size, output_size):
        l = {}
        l['W'] = self.model.add_parameters((in_size, output_size), init=self.init)
        l['b'] = self.model.add_parameters((1, output_size), init=self.init)
        return l

    def one_pass(self, datum):
        datum = dynet.inputTensor(datum)

        w1 = dynet.parameter(self.layers[0]['W'])
        b1 = dynet.parameter(self.layers[0]['b'])
        w2 = dynet.parameter(self.layers[1]['W'])
        b2 = dynet.parameter(self.layers[1]['b'])

        hidden = (datum * w1) + b1
        hidden_activation = dynet.logistic(hidden)
        output = (hidden_activation * w2) + b2
        output_activation = dynet.softmax(output[0])
        #print(output_activation.npvalue())
        #sys.exit()    
        return output_activation


if __name__ == '__main__':
    
    (X,y), (test_X, test_y) = load_mnist(rl = True, pick = False)
    print("done loading")
    print("test x,y")
    print(X.shape, y.shape)
    print("train x,y")
    print(test_X.shape, test_y.shape)

    m = cm(input_size = X.shape[1], output_size = y.shape[1])
    last_loss = None
    last_acc = None
    #X = X[:10000]
    for i in range(100):
        print(i)
        losses = []
        batch_size = 1000
        dynet.renew_cg()
            
        for j in tqdm(range(int(X.shape[0]/batch_size) + 1)):
            for k in range(batch_size):
                index = (j*batch_size) + k
                if index > X.shape[0]-1: break
            # prepare input
                little_x = X[index].reshape(1,-1)   # must make it a vector with dimensions (1 x voc_size)
            # prepare output
                little_y = dynet.inputTensor(y[index])
            # make a forward pass
                pred = m.one_pass(little_x)
            # calculate loss for each example
                loss = dynet.binary_log_loss(pred, little_y)
                if not numpy.isnan(loss.npvalue()):
                    losses.append(loss)
                else:
                    print(i,j,'loss was nan!')
            total_loss = dynet.esum(losses)
        # apply the calculations of the computational graph
            total_loss.forward()
        # calculate loss to backpropogate
            total_loss.backward()
        # update parameters with backpropogated error
            m.trainer.update()
        #### end of batch
        if last_loss:
            cur_loss = total_loss.npvalue()[0]
            print('cur loss:', cur_loss)
            print('\tdif:', cur_loss - last_loss)
            last_loss = cur_loss
        else:
            last_loss = total_loss.npvalue()[0]
        
        if i % 2 == 0:
            test_size = test_X.shape[0]
            correct = 0
            dynet.renew_cg()        
            for j in range(test_size):
                little_x = test_X[j].reshape(1,-1)
                gold_y = numpy.argmax(test_y[j])
                pred = m.one_pass(little_x)
                y_hat = numpy.argmax(pred.npvalue())
                #print(gold_y, y_hat)
                if gold_y == y_hat:
                    correct += 1
            cur_acc = correct / test_size
            if last_acc:
                print('acc:', cur_acc)
                print('\tdif:', cur_acc - last_acc)
                last_acc = cur_acc
            else:
                last_acc = cur_acc
                
