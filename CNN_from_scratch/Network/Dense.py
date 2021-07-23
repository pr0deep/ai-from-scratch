import numpy as np
from Network import activations,utils

class Dense:
    def __init__(self, p, input= (10,784)):
        self.hyper_parameters = {'alpha': 0.001, 'beta1': 0.9, 'beta2': 0.99}
        self.prev_layer = p
        self.weights = utils.std(np.random.rand(input[0],input[1]))
        self.weights /= np.max(self.weights)
        self.bias = utils.std((np.random.rand(10,)))
        self.bias /= np.max(self.bias)
        self.rms_weights = np.zeros(shape=(input[0],input[1]))
        self.velocity_weights = np.zeros(shape=(input[0],input[1]))
        self.rms_bias = np.zeros(10)
        self.velocity_bias = np.zeros(10)
        self.cur_layer = None
        self.output = None
        self.weight_updates = np.zeros_like(self.weights)
        self.bias_updates = np.zeros_like(self.bias)

        #accuracy var
        self.batch_cost = 0

    def clear(self):
        self.velocity_weights = np.zeros_like(self.velocity_weights)
        self.rms_weights = np.zeros_like(self.rms_weights)
        self.rms_bias = np.zeros(10)
        self.velocity_bias = np.zeros(10)
        self.output = None
        self.bias_updates = np.zeros_like(self.bias)
        self.weight_updates = np.zeros_like(self.weights)

    def forward_pass(self):
        prev = self.prev_layer.output.flatten()
        self.cur_layer = np.dot(self.weights,prev)+self.bias
        self.cur_layer -= max(self.cur_layer)
        self.output = activations.softmax(self.cur_layer)

    def backward_pass(self,label):
        y = np.zeros(10)
        y[label] = 1

        #error initialised
        error = np.copy(self.output)
        error[label] -= 1
        #cost calculation for display
        self.batch_cost += utils.cost(y,self.output)

        flat_layer = self.prev_layer.output.flatten()
        #update variables
        for i in range(len(error)):
            for j in range(len(flat_layer)):
                self.weight_updates[i][j] += error[i]*flat_layer[j]
        self.bias_updates += error

        #calculating next error
        next_layer_error = np.zeros_like(flat_layer)

        for i in range(len(flat_layer)):
            for j in range(len(self.weights)):
                next_layer_error[i] += self.weights[j][i]*error[j]

        next_layer_error = next_layer_error.reshape(self.prev_layer.output.shape[0],self.prev_layer.output.shape[1])

        self.prev_layer.backward_pass(next_layer_error)

    def update(self,iteration,BATCH_SIZE):
        #start to calculate velocity and rms
        #weights
        self.velocity_weights = self.hyper_parameters['beta1']*self.velocity_weights + (1-self.hyper_parameters['beta1'])*self.weight_updates/BATCH_SIZE
        self.rms_weights = self.hyper_parameters['beta2'] * self.rms_weights + (1 - self.hyper_parameters['beta2']) * self.weight_updates/BATCH_SIZE*self.weight_updates/BATCH_SIZE
        #biases
        self.velocity_bias = self.hyper_parameters['beta1'] * self.velocity_bias + (
                    1 - self.hyper_parameters['beta1']) * self.bias_updates/BATCH_SIZE
        self.rms_bias = self.hyper_parameters['beta2'] * self.rms_bias + (
                    1 - self.hyper_parameters['beta2']) *  self.bias_updates/BATCH_SIZE * self.bias_updates/BATCH_SIZE

        #biasing the vectors
        v_cap_weights = self.velocity_weights/(1-np.power(self.hyper_parameters['beta1'],iteration))
        r_cap_weights = self.rms_weights / (1 - np.power(self.hyper_parameters['beta2'], iteration))
        v_cap_bias = self.velocity_bias / (1 - np.power(self.hyper_parameters['beta1'], iteration))
        r_cap_bias = self.rms_bias / (1 - np.power(self.hyper_parameters['beta2'], iteration))

        eta = 0.000000001
        #finally updating weights and biases
        self.weights -= (self.hyper_parameters['alpha']*v_cap_weights/(np.sqrt(r_cap_weights+eta)))
        self.bias -= (self.hyper_parameters['alpha']*v_cap_bias/(np.sqrt(r_cap_bias+eta)))
        #self.weights -= (self.hyper_parameters['alpha'] * self.weight_updates/BATCH_SIZE)
        #self.bias -= (self.hyper_parameters['alpha']*self.bias_updates/BATCH_SIZE)
        #preparing for batches
        self.weight_updates = np.zeros_like(self.weight_updates)
        self.bias_updates = np.zeros_like(self.bias_updates)

        self.batch_cost = 0
        return

