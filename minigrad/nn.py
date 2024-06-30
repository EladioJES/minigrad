import random
from minigrad.engine import Value
import matplotlib.pyplot as plt

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params=[]
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

    def update_params(self, learning_rate:float = 0.05):
        for p in self.parameters():
            p.data += -learning_rate * p.grad

    def training_loop(self, X, y, epochs = 10, learning_rate = 0.05):
        print(epochs)
        epoch_loss = []
        for i in range(epochs):
            ypred = [self.__call__(x) for x in X]
            loss = sum([(yout-ygt)**2 for ygt, yout in zip(y, ypred)])
            epoch_loss.append(loss.data)
            if i%10 ==0:
                print(f'Epoch {i} | Loss {loss.data}')
            self.zero_grad()
            loss.backward()
            self.update_params(learning_rate)
        print(f'Epoch {i} | Loss {loss}')

        plt.plot(epoch_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # return epoch_loss
        