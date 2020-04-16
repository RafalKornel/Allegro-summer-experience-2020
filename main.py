#!/usr/bin/python3.7

import numpy as np
from model import Model
from parser import Parser
from matplotlib import pyplot as pl

def sigmoid(x): return 1/(1+np.e**(-x))
def cost(y, label): return -label*np.log(sigmoid(y)) - (1-label)*np.log(1-sigmoid(y))

mnist_train = Parser('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
mnist_valid = Parser('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

network         = [784, 64, 10, 1]
batch           = 300
learning_rate   = 3
iterations      = 100
momentum        = 0.9

model = Model(network, sigmoid, cost)
model.train(mnist_train.data, batch, learning_rate, iterations, momentum, valid=mnist_valid.data)

v, c = model.validate(mnist_valid.data)
print(f"Accuracy and avg. cost of model over whole validation set: {v}, {c}")

name = f"{str(network).replace(' ', '')}_b{batch}_lr{learning_rate}_i{iterations}_m{momentum}_acc{v:.5}"
with open(name, "wb") as f:
    model.serialize(f)

'''
name = "[784,64,10,5,1]_b300_lr3_i100_m0.9_acc0.96424"
recovered_model = Model(network, sigmoid, cost, filename=name)
v, c = recovered_model.validate(mnist_valid.data)
print(f"Accuracy and avg. cost of model over whole validation set: {v}, {c}")
'''