# Author : Pradeep S
# github : github.com/dunder-pradeep


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# to supress that annoying np overflow warning
np.warnings.filterwarnings('ignore', 'overflow')

# some plot init
fig, ax = plt.subplots()

# loading data
train_df = pd.read_csv('./data/train.csv', nrows=3000)
print('train df : ')
print(train_df.head())
train_data = (train_df.to_numpy())

test_df = pd.read_csv('./data/test.csv', nrows=100)
test_data = test_df.to_numpy()


# some functions you may know
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# stable ver of softmax
def softmax(x):
    y = x - np.max(x)
    y = np.exp(y)
    return y / np.sum(y)


# Model definition
class Model:
    def __init__(self, data, split, feature_count, update_method, train_test_plot=False, show_all_plots=False):
        self.layers = []
        self.layers.append(InputLayer(data.shape[1] - 1, update_method))

        splt = int(split * data.shape[0])
        self.train_data = data[0:splt, 1:]
        self.test_data = data[splt:, 1:]
        self.train_true_class = data[:splt, 0]
        self.test_true_class = data[splt:, 0]

        self.update_method = update_method
        self.performance_vars = {'train_costs': [], 'epoch_costs': [], 'train_acc': [], 'test_acc': []}
        self.memory = {'epochs': 0,
                       'features': feature_count, 'data_points': data.shape[0],
                       'split': split, 'cur_it': 0}  # to store some params based on prev training or testing.
        self.args = {'train_test_plot': train_test_plot, 'show_all_plots': show_all_plots}

    def train(self, epochs):
        print(self)
        print(f'\nTraining Model : ')
        self.memory['epochs'] = epochs
        for it in range(epochs):
            # storing iter
            self.memory['cur_iter'] = it
            # shuffling set before each epoch
            self.shuffle()
            # printing iter count
            print('-' * 50)
            print(f'Iter({it + 1}/{epochs}):')
            # adding epoch costs for this iter
            self.performance_vars['epoch_costs'].append(0)

            # training through all the points .
            for i in range(self.train_data.shape[0]):
                self.layers[0].set_input(self.train_data[i])
                self.feed_forward()
                self.backprop(self.train_true_class[i])
                self.update()

                # metrics update per iter
                self.eval_iter_metrics(self.train_true_class[i])
            # metrics update per epoch
            self.eval_epoch_metrics()
        # direct to metrics to take care of plots and other shit
        self.metrics()

    # layer sequential add
    def add_hidden_layer(self, n):
        temp_layer = Layer(n, self.update_method)

        temp_layer.prev = self.layers[-1]
        self.layers[-1].next = temp_layer

        temp_layer.init_weights()
        self.layers.append(temp_layer)

    def add_output_layer(self, n):
        temp_layer = OutputLayer(n, self.update_method)

        temp_layer.prev = self.layers[-1]
        self.layers[-1].next = temp_layer

        temp_layer.init_weights()
        self.layers.append(temp_layer)

    # shuffle dataset
    def shuffle(self):
        perm = np.random.permutation(self.train_data.shape[0])
        self.train_data = self.train_data[perm]
        self.train_true_class = self.train_true_class[perm]

        perm = np.random.permutation(self.test_data.shape[0])
        self.test_data = self.test_data[perm]
        self.test_true_class = self.test_true_class[perm]

    # basic functions to recursively call layer methods
    def feed_forward(self):
        self.layers[0].forward()

    def backprop(self, true_label):
        self.layers[-1].backprop(true_label)

    def update(self):
        self.layers[0].update()

    # evaluation on test set
    def test(self, test_data):
        n = 7
        fig, ax = plt.subplots(n,n)
        fig.suptitle("Test Results")
        [axi.set_axis_off() for axi in ax.ravel()]
        for i in range(n):
            for j in range(n):
                ax[i][j].set_axis_off()
                pred = self.predict(test_data[n*i+j])
                ax[i][j].set_title(str(pred))

                ax[i][j].imshow(test_data[i*n+j].reshape((28, 28)), cmap='gray')
        plt.tight_layout()
        plt.show()

    def predict(self, sample):
        self.layers[0].set_input(sample)
        self.feed_forward()
        return np.argmax(self.layers[-1].output)

    # metrics
    def metrics(self):
        if self.args['show_all_plots']:
            self.plot_all_metrics()
            plt.show()
        elif self.args['train_test_plot']:
            self.plot_train_test()
            plt.show()

    def eval_iter_metrics(self, true_class):
        it_cost = self.cost(true_class)
        self.performance_vars['train_costs'].append(it_cost)
        self.performance_vars['epoch_costs'][self.memory['cur_it']] += it_cost

    def eval_epoch_metrics(self):
        self.calc_test()
        self.calc_train()

    def cost(self, true_class):
        return -np.log(0.0001 + self.layers[-1].output[true_class])

    def calc_test(self):
        correct = 0
        for idx in range(self.test_data.shape[0]):
            self.layers[0].set_input(self.test_data[idx])
            self.feed_forward()
            if np.argmax(self.layers[-1].output) == self.test_true_class[idx]:
                correct += 1
        self.performance_vars['test_acc'].append(correct / self.test_data.shape[0])
        print(f'Test Accuracy : {correct / self.test_data.shape[0]} ({correct}/{self.test_data.shape[0]})')

    def calc_train(self):
        correct = 0
        for idx in range(int(0.1 * self.train_data.shape[0])):
            self.layers[0].set_input(self.train_data[idx])
            self.feed_forward()
            if np.argmax(self.layers[-1].output) == self.train_true_class[idx]:
                correct += 1
        self.performance_vars['train_acc'].append(correct / int(0.1 * self.train_data.shape[0]))
        print('Train Acc: ', correct / int(0.1 * self.train_data.shape[0]))

    def plot_all_metrics(self):
        fig, ax = plt.subplots(1, 2)
        self.plot_train_test(ax[0])
        self.plot_cost_per_epoch(ax[1])

    def plot_cost_per_iter(self, ax=ax):
        ax.set_title('cost( -log(y) )')
        sb.lineplot(x=list(range(self.memory['epochs'] * self.train_data.shape[0])),
                    y=self.performance_vars['train_costs'], ax=ax)

    def plot_cost_per_epoch(self, ax=ax):
        ax.set_title('cost per epoch ( formula : -log(y) )')
        sb.lineplot(x=list(range(self.memory['epochs'])), y=self.performance_vars['epoch_costs'], ax=ax)

    def plot_train_test(self, ax=ax):
        ax.set_title('Train - Test accuracy per epoch')
        sb.lineplot(x=list(range(self.memory['epochs'])), y=self.performance_vars['train_acc'], ax=ax)
        sb.lineplot(x=list(range(self.memory['epochs'])), y=self.performance_vars['test_acc'], ax=ax)
        ax.legend(['train', 'test'])

    # modifying str
    def __str__(self):
        return f"\nModel : {{   \n Features : {self.memory['features']} \n Weight initialise : N(0,1)/sqrt(n) \n Bias initialise : " \
               f"zeros  \n Data points : {self.memory['data_points']} \n" \
               f" Split : {self.memory['split']} \n" \
               f" layer_count : {len(self.layers)}\n" \
               f" update method : {self.update_method} \n }}"


# A hidden layer class with generic properties and methods.
class Layer:
    def __init__(self, n, update_method):
        self.update_method = update_method
        self.weights = None
        self.bias = None
        self.next = None
        self.prev = None
        self.output = None
        self.weight_node_count = n
        self.weight_gradients = None
        self.bias_gradients = None
        self.momentum_history_weights = 0
        self.momentum_history_bias = 0
        self.error = None
        # params : alpha ->learning rate, beta-> momentum,gamma->Adam
        self.params = {'alpha': 0.001, 'beta': 0.9}

    # method to initialise weights.
    def init_weights(self):
        if self.prev is None:
            print('failed to init weights at a layer')
        else:
            self.weights = np.array([np.random.normal(size=self.prev.weight_node_count) for _ in
                                     range(self.weight_node_count)],
                                    dtype=np.float64)/np.sqrt(self.weight_node_count)
            self.weight_gradients = np.zeros_like(self.weights)
            self.bias = np.random.normal(size=self.weight_node_count)
            self.bias_gradients = np.zeros_like(self.bias)

    # basic feed forward dot product ..
    def forward(self):
        self.output = sigmoid(self.bias + self.weights @ self.prev.output)
        if self.next is None:
            return
        else:
            self.next.forward()

    # backprop
    def backprop(self, true_label):
        # backprop code
        if type(self).__name__ == 'OutputLayer':
            err = self.output.copy()
            err[true_label] = err[true_label] - 1
            self.bias_gradients = err
            self.weight_gradients = err.reshape(err.shape[0], 1) * self.prev.output
            self.error = err
        else:
            self.error = self.next.weights.T @ self.next.error
            self.bias_gradients = (self.error * self.output * (1 - self.output))
            self.weight_gradients = self.bias_gradients.reshape((self.weight_node_count, 1)) * self.prev.output
        # traverse code
        if self.prev is None:
            return
        else:
            self.prev.backprop(true_label)

    # layer weights update code
    def update(self):
        update_method = self.update_method
        if update_method == 'normal':
            self.weights -= self.params['alpha'] * self.weight_gradients
            self.bias -= self.params['alpha'] * self.bias_gradients

        if update_method == 'momentum':
            cur_weight_update = self.params['alpha'] * self.weight_gradients + self.params[
                'beta'] * self.momentum_history_weights  # adding the momentum history
            self.weights -= cur_weight_update
            self.momentum_history_weights = cur_weight_update

            cur_bias_update = self.params['alpha'] * self.bias_gradients + self.params[
                'beta'] * self.momentum_history_bias
            self.bias -= cur_bias_update
            self.momentum_history_bias = cur_bias_update

        if self.next is None:
            return
        else:
            self.next.update()


# A layer associated with the start of the model. called while initialising model.
class InputLayer(Layer):
    def __init__(self, n, update_method):
        super(InputLayer, self).__init__(n, update_method)
        self.weights = None
        self.weight_node_count = n

    def forward(self):
        if self.next is None:
            print('Empty model')
            return
        else:
            # do nothing with prev layer
            self.next.forward()

    def set_input(self, data):
        self.output = data

    def backprop(self, _):
        return

    def update(self):
        self.next.update()


# A densely connected output layer which has a diff output func.
class OutputLayer(Layer):
    def __init__(self, n, update_method):
        super(OutputLayer, self).__init__(n, update_method)
        self.weight_node_count = n

    def forward(self):
        self.output = softmax(self.bias + self.weights @ self.prev.output)


model = Model(train_data, split=0.8, feature_count=int(784), update_method='momentum', show_all_plots=True,
              train_test_plot=False)
model.add_hidden_layer(128)
model.add_output_layer(10)
model.train(epochs=5)

model.test(test_data)
