import numpy as np
import pickle
from sympy import *
from sympy.abc import x
import copy
import random
from matplotlib import pyplot as pl

def timer(fun):

    def inner(*args):
        t0 = time.time()
        result = fun(*args)
        t1 = time.time()

        print(f"I did {fun.__name__} and it took me: {(t1 - t0) * 1000} miliseconds")
        return result

    return inner

class Model():
    def __init__(self, layers, activation_function, cost_function, filename=""):

        self.layers = layers
        self.act = activation_function
        self.cost = cost_function
        self.act_der = lambdify(x, diff(self.act(x), x))

        if filename != "":
            with open(filename, "rb") as f:
                self.l = pickle.load(f)
                self.b = pickle.load(f)
                self.z = pickle.load(f)
                self.d = pickle.load(f)
                self.w = pickle.load(f)
        else:
            self.l = np.array([ np.array([ 0.0 for _ in range(i) ]) for i in layers ])
            self.b = np.array([ np.array([random.random() for x in range(len(self.l[i])) ]) \
                                                    for i in range(len(self.l))])
            self.z = copy.deepcopy(self.l)
            self.d = copy.deepcopy(self.z)
            self.w = [ np.random.uniform(low=-1.0, high=1.0, size=(len(self.l[i+1]), \
                            len(self.l[i]))) for i in range(len(self.l) - 1) ]


    def validate(self, valid_data):
        valid = 0
        cost = 0
        for i in range(len(valid_data)):
            predicted = self.predict(valid_data[i][0], verbose=False)
            classification = 1 if predicted > 0.5 else 0
            expected = valid_data[i][1] * 1.

            if classification == expected: valid += 1
            cost += self.cost(predicted, valid_data[i][1])

        return valid/len(valid_data)*1., cost/len(valid_data) * 1.



    def predict(self, data_entry, verbose=False):
        self.l[0] = data_entry
        self.z[0] = data_entry

        for i in range(1, len(self.l)):

            self.z[i] = np.dot(self.w[i - 1], self.l[i - 1]) + self.b[i]
            self.l[i] = self.act(self.z[i])
        
        if verbose: 
            s = ''
            for i, x in enumerate(self.l[-1]): s += f'{i+1:0}: {x:.2}   '
            print(f"I predicted these values:\t{s}")
        return self.l[-1]


    def backpropagation(self, inp, label):
        for i in range(len(self.l)-1 , -1, -1):

            self.predict(inp, verbose=False)
            if i == len(self.l) - 1:
                self.d[i] = ( - label*1./(1. + np.exp(self.z[i])) + (1.-label)*1./(1+np.exp(-self.z[i])) ) * self.act_der(self.z[i])
            else:
                self.d[i] = np.dot(np.transpose(self.w[i]), self.d[i + 1]) * self.act_der(self.z[i])
            
        db = self.d
        dw = [ np.array([ self.l[j] * self.d[j+1][i] for i in range(len(self.d[j+1]))]) for j in range(len(self.w)) ]

        return dw, db

    
    def train(self, data, batch_size=1, step_size=0.1, iterations=1000, beta=0.9, debug=False, **kwargs):
        print(batch_size, step_size, iterations, beta, self.layers)
        acc = []
        cost = []
        c = step_size/batch_size

        for i in range(iterations):
            batch = random.sample(data, batch_size)

            Vw = [ np.zeros(w.shape, dtype=np.float64) for w in self.w]
            Vb = np.zeros(self.b.shape, dtype=np.float64)

            for e in batch:
                dw, db = self.backpropagation(e[0], e[1])
                for j in range(len(Vw)):
                    Vw[j] = beta * Vw[j] + (1 - beta) * dw[j]
                Vb = beta * Vb + (1 - beta) * db
                self.w = [ self.w[j] - Vw[j] * c for j in range(len(dw))]
                self.b = self.b - Vb*c
            

            if "valid" in kwargs:
                valid_batch = random.sample(kwargs["valid"], batch_size)
                v, c = self.validate(valid_batch)
                acc.append(v)
                cost.append(c)
                print(f"Interation: {i + 1} \t\tAccuracy: {v}")

        if len(acc) != 0:
            _dpi = 96
            pl.figure(figsize=(1280/_dpi, 720/_dpi), dpi=_dpi)
            pl.plot(acc, 'b', linewidth=1.4, label="Model accuracy")
            pl.plot(cost, 'r', linewidth=1.4, label="Avg. cost over batch")
            pl.title(f"Accuracy and cost over (batches of) validation set. Batch:{batch_size}, Learning rate:{step_size}, Iterations:{iterations}, Momentum beta:{beta}")
            pl.legend()
            pl.text(int(iterations*0.85), 0.75, f"Network structure: \n{self.layers}")
            pl.xlabel("Iteration")
            pl.ylabel("Value")
            name = f"{str(self.layers).replace(' ', '')}_b{batch_size}_lr{step_size}_i{iterations}_m{beta}.png"
            pl.savefig(name, dpi=_dpi)



    def serialize(self, filename):
        pickle.dump(self.l, filename)
        pickle.dump(self.b, filename)
        pickle.dump(self.z, filename)
        pickle.dump(self.d, filename)
        pickle.dump(self.w, filename)


    def __str__(self):
        sep = "\n"
        s = f"\nWeights: {type(self.w)} \n{sep.join(str(_w) for _w in self.w)} \n\nBias: \
        {type(self.b)} \n{sep.join(str(_b) for _b in self.b)} \n\nLayers: \
        {type(self.l)}\n{sep.join(str(_l) for _l in self.l)}\n\nZ: \
        {type(self.z)}\n{sep.join(str(_z) for _z in self.z)}\n"
        return s