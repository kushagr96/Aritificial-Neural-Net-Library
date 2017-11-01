import numpy as np
import matplotlib.pyplot as plt

class DenseNet:
    def __init__(self, input_dim, optim_config, loss_fn):
        """
        Initialize the computational graph object.
        """
        self.graph = Graph(input_dim, optim_config, loss_fn)
        pass
        
    def addlayer(self, activation, units):
        """
        Modify the computational graph object by adding a layer of the specified type.
        """
        self.graph.addgate(activation, units)
        pass

    def train(self, X, Y):
        """
        This train is for one iteration. It accepts a batch of input vectors.
        It is expected of the user to call this function for multiple iterations.
	"""
	i = np.random.randint(0,len(X))
	self.graph.forward(X[i])
        loss_value = self.graph.backward(Y[i])
        self.graph.update()
        return (np.abs(loss_value))

    def predict(self, X):
        """
        Return the predicted value for all the vectors in X.
        """
	return self.graph.forward(X)

class Graph:
    # Computational graph class
    def __init__(self, input_dim, optim_config, loss_fn):
        self.layers_dim = [input_dim]
        self.optim_config = optim_config
        self.W = []
        self.b = []
        self.activation_list = []
	self.W_change_store_new = []
	self.b_change_store_new = []
	self.W_change_store_old = []
	self.b_change_store_old = []
        self.i = 0
        self.loss_fn = loss_fn
        pass

    def addgate(self, activation, units=0):
        if activation.lower() == 'relu':
            activation_fn = ReLU()
        elif activation.lower() == 'sigmoid':
            activation_fn = Sigmoid()
        elif activation.lower() == 'softmax':
            activation_fn = Softmax()
        else:
            activation_fn = Linear()
        self.activation_list.append(activation_fn)
        self.layers_dim.append(units)
        self.W.append( 2 * np.random.random_sample((self.layers_dim[self.i], self.layers_dim[self.i+1])) - 1)
        self.b.append((2 * np.random.random(self.layers_dim[self.i+1]) - 1).reshape(1, self.layers_dim[self.i+1]))  
        self.i += 1  
        pass

    def forward(self, input):
        self.layer_value = []
        self.deltas = []
        self.input = input.reshape(1,self.layers_dim[0])
        self.layer_value.append(self.input)
        fn_output = self.input
        j = 0
        for activation_fn in self.activation_list:
            adder = np.dot(fn_output, self.W[j]) + self.b[j]
            fn_output = activation_fn.forward(adder)
            self.layer_value.append(fn_output) 
            j += 1
	self.W_change_store_old = self.W_change_store_new
	self.b_change_store_old = self.b_change_store_new
	self.W_change_store_new = []
	self.b_change_store_new = []	
        return fn_output

    def backward(self, expected):
        self.expected = expected
        D = expected
        Y = self.forward(self.input)
        if self.loss_fn.lower() == 'l1loss':
            loss = L1Loss(D, Y)
        elif self.loss_fn.lower() == 'l2loss':
            loss = L2Loss(D, Y)
        elif self.loss_fn.lower() == 'crossentropy':
            loss = CrossEntropy(D, Y)
        else:
            loss = SVMloss(D, Y)
        loss_value = loss.loss()
        loss_deriv = loss.deriv()
        k = len(self.W) - 1
        dz = loss_deriv
        for activation_fn in reversed(self.activation_list):
            layer_delta = activation_fn.backward(dz) 
            self.deltas.append(layer_delta)
            dz = np.dot(layer_delta, np.transpose(self.W[k]))
            k-=1
	self.deltas.reverse()
        return loss_value

    def update(self):
        j = len(self.layers_dim) - 2
        for k in reversed(xrange(len(self.W))):	
	    	if len(self.W_change_store_old) < len(self.W):	
		    W_change = optim_config.eta * np.dot( np.transpose(self.layer_value[j]), self.deltas[k])
		    b_change = optim_config.eta * self.deltas[k]
		else:
		    W_change = optim_config.eta * np.dot( np.transpose(self.layer_value[j]), self.deltas[k]) + optim_config.mu * self.W_change_store_old[len(self.W) - k -1]
		    b_change = optim_config.eta * self.deltas[k] + optim_config.mu * self.b_change_store_old[len(self.W) - k -1]		    	
		self.W[k] -= W_change
	        self.b[k] -= b_change
	        self.W_change_store_new.append(W_change)
	        self.b_change_store_new.append(b_change)	
	        j-=1		
        pass


class Optimizer:
    def __init__(self, learning_rate, momentum_eta = 0.0):
        self.eta = learning_rate
        self.mu = momentum_eta
        pass

        
class ReLU:
    def __init__(self, d=0, m=0):
        self.d = d
        self.m = m        
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0.0, input)

    def backward(self, dz):
        return 1.0 * (self.input > 0) * dz


class Sigmoid:

    def __init__(self, d=0, m=0):
        self.d = d
        self.m = m        
        pass

    def forward(self, input):
        self.output =  1.0 / (1.0 + np.exp(-input))
        return self.output

    def backward(self, dz):
        return (1.0 - self.output) * self.output * dz


class Softmax:

    def __init__(self, d=0, m=0):
        self.d = d
        self.m = m        
        pass

    def forward(self, input):
        self.output = np.exp(input) / np.sum(np.exp(input))
        return self.output

    def backward(self, dz):
    	l = len(self.output)
    	gradient = np.zeros((l,l))
    	for i in xrange(l):
    		for j in xrange(l):
    			if i == j:
    				gradient[i][j] = self.output[0][i] * (1 - self.output[0][i])
			else:
    				gradient[i][j] = - self.output[0][i] * self.output[0][j]
    	return gradient * dz 

class Linear:

    def __init__(self, d=0, m=0):
        self.d = d
        self.m = m        
        pass

    def forward(self, input):
        self.input = input
        return input

    def backward(self, dz):
        output = self.forward(X)
        return dz


class L1Loss:

    def __init__(self, D, Y):
        self.y = Y
        self.d = D
        pass

    def loss(self):
        return np.sum(np.abs(self.d-self.y))

    def deriv(self):
        return -1.0 * ((self.y - self.d) > 0) + 1.0 * ((self.y - self.d) < 0)


class L2Loss:
    def __init__(self, D, Y):
        self.d = D
        self.y = Y
        pass

    def loss(self):
        return np.sum(np.power(self.d-self.y, 2)) / 2.0

    def deriv(self):
        return self.y-self.d 


class CrossEntropy:
    def __init__(self,D,Y):
        self.d = D
        self.y = Y
        self.l = len(D)
        pass

    def loss(self):
        return -np.sum(self.d*np.log(self.y))

    def deriv(self):
    	if self.l < 3:
    		return (self.y - self.d) / self.y * (1 - self.y)
    	else:
    		return -self.d / self.y 

class SVMloss:
    def __init__(self, D, Y):
        self.d = D
        self.y = Y
	self.output = np.zeros(len(D))
        pass

    def loss(self):
        value = 0.0
        for i in xrange(len(self.d)):
            if self.d[i] > 0:
                yi = i
                break
        for j in xrange(len(self.d)):
            if j == yi:
            	self.output[j] = 0
                continue
            self.output[j] = self.y[0][j] - self.y[0][yi] + 1    
            value += np.maximum(0, self.output[j])
        return value

    def deriv(self):
        return 1.0 * (self.output > 0)

