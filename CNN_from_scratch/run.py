from Network import model,utils
import mnist,time,matplotlib.pyplot as plt,matplotlib as mpl


model.add("Input")
model.add("Convolution",kernel_size=5,layer_size=(24,24))
model.add("Convolution",kernel_size=5,layer_size=(20,20))
model.add("Dense",output_size=(10,20*20))

data = mnist.train_images()
labels = mnist.train_labels()

EPOCHS = 3
BATCH_SIZE = 20
IMAGES = 50000
BATCHES = int(IMAGES/BATCH_SIZE)
TEST_IMAGE_START = IMAGES
TEST_IMAGE_NO = 20
#plot data
batch_cost = []
batch_train_ac = []
batch_test_ac = []


for epoch in range(EPOCHS):
    start_time = time.time()
    for batch in range(BATCHES):
        b_cost,tr,te = model.mini_batch(data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE],labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE],BATCH_SIZE,epoch+1,data[TEST_IMAGE_START:TEST_IMAGE_START+TEST_IMAGE_NO],labels[TEST_IMAGE_START:TEST_IMAGE_START+TEST_IMAGE_NO],batch+1,BATCHES)
        batch_cost.append(b_cost)
        batch_train_ac.append(tr)
        batch_test_ac.append(te)
    end_time = time.time()
    model.clear()
    train_ac = model.test(data[batch*BATCH_SIZE:(batch)*BATCH_SIZE+10],labels[batch*BATCH_SIZE:(batch)*BATCH_SIZE+10])
    test_ac = model.test(data[TEST_IMAGE_START:TEST_IMAGE_START+TEST_IMAGE_NO],
                          labels[TEST_IMAGE_START:TEST_IMAGE_START+TEST_IMAGE_NO])
    model.clear()
    print("Epoch (",epoch+1,"/",EPOCHS,") time: ",end_time-start_time,"s train_ac : ",train_ac*100,"% test_ac : ",test_ac*100,"% ")
    print()


#data validation
utils.model_visualise(EPOCHS,BATCHES,batch_cost,batch_train_ac,batch_test_ac)


