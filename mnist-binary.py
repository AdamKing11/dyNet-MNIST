import os, sys, re
import numpy
import csv
import pickle
from tqdm import tqdm
from pprint import pprint

from sklearn.metrics import log_loss

import dynet_config
dynet_config.set(
    mem=2048,          # can probably get away with 1024
    autobatch=True,    # utilize autobatching
    random_seed=1978   # simply for reproducibility here
)
import dynet

y = dynet.inputTensor([.2, .3, .5])


print(y.npvalue())


sys.exit()

# 0 vs 1
def load(f = 'mnist_train.csv'):
    x_py = []
    y_py = []
    with open(f) as rf:
        reader = csv.reader(rf, delimiter = ',')
        for line in tqdm(reader):
            if line[0] in ['0', '1']:
                y_py.append([line[0]])
            #y_onehot = [0 for _ in range(10)]
            #y_onehot[int(line[0])] = 1
            #y_py.append(y_onehot)
                x_py.append(line[1:])
    return numpy.asarray(x_py, dtype = 'int8'), numpy.asarray(y_py, dtype = 'int8')

print('training data....')
try:
    #assert 1 == 0
    X = pickle.load(open('train_X.pick', 'rb'))
    y = pickle.load(open('train_y.pick', 'rb'))
except:
    X,y = load()
    pickle.dump(X, open('train_X.pick', 'wb'))
    pickle.dump(y, open('train_y.pick', 'wb'))
print('testing data....')
try:
    #assert 1 == 0
    test_X = pickle.load(open('test_X.pick', 'rb'))
    test_y = pickle.load(open('test_y.pick', 'rb'))
except:
    test_X,test_y = load('mnist_test.csv')
    pickle.dump(test_X, open('test_X.pick', 'wb'))
    pickle.dump(test_y, open('test_y.pick', 'wb'))



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
        self.trainer = dynet.SimpleSGDTrainer(m=self.model, learning_rate=.01)


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
        output = (hidden * w2) + b2
        output_activation = dynet.logistic(output)    
        return output_activation


if __name__ == '__main__':
    
    print("done loading")
    print("test x,y")
    print(X.shape, y.shape)
    print("train x,y")
    print(test_X.shape, test_y.shape)

    m = cm(input_size = X.shape[1], output_size = y.shape[1])
    last_loss = None
    last_mse = None
    for i in range(100):
        print(i)
        dynet.renew_cg()
        losses = []
        for j in tqdm(range(X.shape[0])):
            # prepare input
            little_x = X[j].reshape(1,-1)   # must make it a vector with dimensions (1 x voc_size)
            
            # prepare output
            little_y = dynet.scalarInput(y[j][0])
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
        if last_loss:
            cur_loss = total_loss.npvalue()[0]
            print('cur loss:', cur_loss)
            print('\tdif:', last_loss - cur_loss)
            last_loss = cur_loss
        else:
            last_loss = total_loss.npvalue()[0]
        
        errors = []
        dynet.renew_cg()        
        for j in range(1000):
            little_x = test_X[j].reshape(1,-1)
            gold_y = test_y[j][0]
            pred = m.one_pass(little_x)
            mse = (gold_y - pred.npvalue()[0]) ** 2
            errors.append(mse)
        if last_mse:
            cur_mse = sum(errors)
            print('mse:', cur_mse/len(errors))
            print('\tdif:', (last_mse - cur_mse)/len(errors))
            last_mse = cur_mse
        else:
            last_mse = sum(errors)