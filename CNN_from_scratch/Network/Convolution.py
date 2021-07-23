import numpy as np
from Network import activations,utils

class Convolution:
    def __init__(self, p, n, kernel_size=5, layer_size=(10, 10)):
        self.hyper_parameters = {'alpha': 0.001, 'beta1': 0.9, 'beta2': 0.99}
        self.layer = None
        self.prev_layer = p
        self.next_layer = n
        self.kernel_size = kernel_size
        self.kernel = utils.std(np.random.rand(kernel_size, kernel_size))
        self.kernel /= np.max(self.kernel)
        self.bias = utils.std(np.random.rand(layer_size[0],layer_size[1]))
        self.bias /= np.max(self.bias)
        self.output = None
        self.velocity_kernel = np.zeros(shape=(kernel_size, kernel_size))
        self.rms_kernel = np.zeros(shape=(kernel_size, kernel_size))
        self.velocity_bias = np.zeros(shape=(layer_size[0],layer_size[1]))
        self.rms_bias = np.zeros(shape=(layer_size[0],layer_size[1]))
        self.kernel_update = np.zeros(shape=(kernel_size, kernel_size))
        self.bias_update = np.zeros(shape=(layer_size[0],layer_size[1]))
        self.relu_der = None

    def clear(self):
        self.velocity_kernel = np.zeros_like(self.velocity_kernel)
        self.rms_kernel = np.zeros_like(self.rms_kernel)
        self.velocity_bias = np.zeros_like(self.velocity_bias)
        self.rms_bias = np.zeros_like(self.rms_bias)
        self.kernel_update = np.zeros_like(self.kernel_update)
        self.bias_update = np.zeros_like(self.bias_update)

    def forward_pass(self):

        prev = self.prev_layer.output
        l = len(prev) - len(self.kernel) + 1
        self.layer = np.zeros(shape=(l, l))
        k = self.kernel_size

        # convolution
        for i in range(l):
            for j in range(l):
                self.layer[i][j] = np.dot(self.kernel.flatten() , prev[i:i + k,j:j + k].flatten())

        #passing to relu
        self.output,self.relu_der = activations.relu(self.layer+self.bias)
        self.next_layer.forward_pass()
    def backward_pass(self, error):
        #error propogated
        k = len(self.kernel)

        error *= self.relu_der
        self.relu_der = None

        next_layer_error = np.zeros_like(self.prev_layer.output)

        for i in range(len(error)):
            for j in range(len(error)):
                self.kernel_update += error[i][j]*self.prev_layer.output[i:i+k,j:j+k]
                next_layer_error[i:i+k,j:j+k] += error[i][j]*self.kernel

        self.bias_update += error

        self.prev_layer.backward_pass(next_layer_error)

    def update(self, iteration,BATCH_SIZE):
        # start to calculate velocity and rms
        # kernel
        self.velocity_kernel = self.hyper_parameters['beta1'] * self.velocity_kernel + (
                    1 - self.hyper_parameters['beta1']) * self.kernel_update/BATCH_SIZE
        self.rms_kernel = self.hyper_parameters['beta2'] * self.rms_kernel + (
                    1 - self.hyper_parameters['beta2']) * self.kernel_update /BATCH_SIZE* self.kernel_update/BATCH_SIZE
        # biases
        self.velocity_bias = self.hyper_parameters['beta1'] * self.velocity_bias + (
                1 - self.hyper_parameters['beta1']) * self.bias_update/BATCH_SIZE
        self.rms_bias = self.hyper_parameters['beta2'] * self.rms_bias + (
                1 - self.hyper_parameters['beta2']) * self.bias_update/BATCH_SIZE * self.bias_update/BATCH_SIZE

        # biasing the vectors
        v_cap_kernel = self.velocity_kernel / (1 - np.power(self.hyper_parameters['beta1'], iteration))
        r_cap_kernel = self.rms_kernel / (1 - np.power(self.hyper_parameters['beta2'], iteration))
        v_cap_bias = self.velocity_bias / (1 - np.power(self.hyper_parameters['beta1'], iteration))
        r_cap_bias = self.rms_bias / (1 - np.power(self.hyper_parameters['beta2'], iteration))

        eta = 0.000000001

        # finally updating kernel and biases
        self.kernel -= (self.hyper_parameters['alpha'] * v_cap_kernel / (np.sqrt(r_cap_kernel + eta)))
        self.bias -= (self.hyper_parameters['alpha'] * v_cap_bias / (np.sqrt(r_cap_bias + eta)))

        #self.kernel -= self.hyper_parameters['alpha']*self.kernel_update/BATCH_SIZE
        #self.bias -= self.hyper_parameters['alpha']*self.bias_update/BATCH_SIZE
        # preparing for batches

        self.bias_update = np.zeros_like(self.bias_update)
        self.kernel_update = np.zeros_like(self.kernel_update)

        self.next_layer.update(iteration,BATCH_SIZE)

