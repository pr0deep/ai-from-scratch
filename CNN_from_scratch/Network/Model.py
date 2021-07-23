from Network import Convolution, Input, Dense


class Model:
    def __init__(self):
        self.layers = []

    def mini_batch(self, image_data, label_data, batch_size, iteration,test_data,test_labels,id,total):
        for i in range(batch_size):
            self.layers[0].forward_pass(image_data[i])
            self.layers[-1].backward_pass(label_data[i])
        c = self.layers[-1].batch_cost/batch_size
        train_ac = self.test(image_data[:5],label_data[:5])
        test_ac = self.test(test_data,test_labels)
        Model.batch_info(id,train_ac,test_ac,c,total)
        self.layers[0].update(iteration, batch_size)
        return c,train_ac*100,test_ac*100

    @staticmethod
    def batch_info(batch,train_ac,test_ac,cost,total):
        print("Batch (",batch,"/",total,"): train_ac: ",train_ac*100,"% test_ac :",test_ac*100,"% cost : ",cost)

    def clear(self):
        for i in self.layers:
            i.clear()

    def add(self, name, kernel_size=5, layer_size=None,output_size=None):

        if name == "Convolution":
            self.layers.append(Convolution.Convolution(self.layers[-1], None, kernel_size, layer_size))
            self.layers[-2].next_layer = self.layers[-1]

        if name == "Dense":

            self.layers.append(Dense.Dense(self.layers[-1], output_size))
            self.layers[-2].next_layer = self.layers[-1]

        if name == "Input":
            self.layers.append(Input.Input(None))

    def test(self, images,labels):
        acc = 0
        for i in range(len(images)):
            self.layers[0].forward_pass(images[i])
            acc += self.layers[-1].output[labels[i]]
        return acc/len(images)