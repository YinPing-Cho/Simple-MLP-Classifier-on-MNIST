import utils
import layers
import network
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(1)

img_rows,img_cols = 28,28
#downsample rate
d_rate = 2

#load training data
train_data = np.load('train.npz')
train_image_data, train_label_data= train_data['image'], train_data['label']

print("Training Images: ",train_image_data.shape)
print("Training Labels: ",train_label_data.shape)

Training_Data = []
for i in range(len(train_image_data)):
    Training_Data.append([train_image_data[i][::d_rate,::d_rate], train_label_data[i]])
np.random.shuffle(Training_Data)

plt.imshow(Training_Data[2][0], cmap = 'gray')
plt.title("Class "+ str(Training_Data[2][1]))
plt.show()
plt.clf()

for train in Training_Data:
    train[0] /= 2550
    chain = itertools.chain(*train[0])
    train[0] = list(chain)
    
    foo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    foo[int(train[1])] = 1
    train[1] = foo

#print(Training_Data[2][0])
print("Label: ", Training_Data[2][1])

#load testing data
test_data = np.load('test.npz')
test_image_data, test_label_data= test_data['image'], test_data['label']

print("Testing Images: ",test_image_data.shape)
print("Testing Labels: ",test_label_data.shape)

Testing_Data = []
for i in range(len(test_image_data)):
    Testing_Data.append([test_image_data[i][::d_rate,::d_rate], test_label_data[i]])
np.random.shuffle(Testing_Data)

plt.imshow(Testing_Data[2][0], cmap = 'gray')
plt.title("Class "+ str(Testing_Data[2][1]))
plt.show()
plt.clf()

for test in Testing_Data:
    test[0] /= 2550
    chain = itertools.chain(*test[0])
    test[0] = list(chain)
    
    foo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    foo[int(test[1])] = 1
    test[1] = foo

#print a sample to make sure data is loaded correctly
print("Label: ", Testing_Data[2][1])

#arguments: (shape, outputlayer, randominit, learningrate, epochs)
whatever = 196
nn = network.MLP([int(784/d_rate/d_rate), int(whatever), int(whatever*0.5), int(whatever*0.5), 2], n_outputs=10, RandomInit=True, LearningRate=0.01, Epochs=20)
nn.Trainer(Training_Data, Testing_Data, Visualize=True)