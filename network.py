import numpy as np
import utils
import layers
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd


class MLP:

    def __init__(self, Shape, n_outputs, RandomInit=True, LearningRate=0.01, Epochs=100):
        self.loss_log = []
        self.testing_log = []
        self.training_log = []
        self.latent = []
        self.iterations = 0
        self.epochs = int(Epochs)
        self.learningrate = LearningRate
        self.hidden = []
        for i in range(0, len(Shape)-1):
            self.hidden.append(layers.Hidden(Shape[i], Shape[i+1], RandomInit))
        self.hidden.append(layers.Hidden(Shape[-1], n_outputs, RandomInit))
        self.outputlayer = layers.Output()
        self.accu = 0
        self.done_epochs = 0

    def activate_pass(self, Input, Truth, Feature, fin):
        self.hidden[0].activate_pass(Input)
        for i in range(1, len(self.hidden)):
            self.hidden[i].activate_pass(self.hidden[i-1].Z)
        self.outputlayer.update(self.hidden[-1].Z, Truth, Feature, fin)
        self.iterations += 1

    def back_propagation(self):
        for i in range(len(self.outputlayer.Grad_pass)):
            self.hidden[-1].update(self.outputlayer.Grad_pass[i], self.learningrate*(max(1-self.accu, (self.done_epochs < self.epochs/2))))
            self.hidden[-1].regularize()
            for j in range(len(self.hidden)-2, -1, -1):
                self.hidden[j].update(self.hidden[j+1].Grad_pass, self.learningrate*(max(1-self.accu, (self.done_epochs < self.epochs/2))))
                self.hidden[j].regularize

        self.outputlayer.Grad_pass.clear()
        
    def Trainer(self, TrainingData, TestingData, Visualize=False):
        print(self.epochs, " Epochs.")
        iters = 0
        for i in range(self.epochs):
            self.done_epochs = i
            epochcount = 0
            if i == int(self.epochs / 10) or i == int(self.epochs * 0.95):
                feature = True
            else: feature = False
            
            curr_test = 0
            curr_train = 0

            np.random.shuffle(TestingData)
            np.random.shuffle(TrainingData)
            accu_train = 0
            for j in range(len(TrainingData)):
                
                if j % 1000 == 0:
                    '''
                    testing
                    '''
                    accu_test = 0
                    
                    for k in range(100):
                        if i >= self.epochs - 19:
                            fin = True
                        else: fin = False
                        self.activate_pass(TestingData[k+int(j/10)][0], TestingData[k+int(j/10)][1], feature, fin)
                        accu_test += self.outputlayer.right
                    curr_test = accu_test / 100
                    self.accu = curr_test
                    self.testing_log.append(self.outputlayer.log[:])
                    self.outputlayer.Grad_pass.clear()
                    self.outputlayer.log.clear()

                '''
                training
                '''
                fin = False
                self.activate_pass(TrainingData[j][0], TrainingData[j][1], feature, fin)
                accu_train += self.outputlayer.right
                self.back_propagation()
                iters += 1
                if iters % 100 == 0:
                    curr_train = accu_train / 100
                    accu_train = 0
                    utils.printProgressBar(iters, self.epochs*len(TrainingData), prefix = 'Progress:', suffix = 'Epoch '+str(i),
                     length = 20, test = curr_test, train = curr_train)
                self.training_log.append(self.outputlayer.log[:])
                
            self.loss_log.append(sum(self.outputlayer.crossentropy_log)/1300)
            self.outputlayer.crossentropy_log.clear()

            if len(self.outputlayer.feature) != 0:
                self.latent.append(self.outputlayer.feature[:])
                self.outputlayer.feature.clear()

        '''
        For plotting results
        '''
        confusion = self.outputlayer.confusion[:]
        cf = pd.DataFrame(confusion)
        cf.to_csv("confusion.csv")
        print("Output done.")

        count = 0
        summ = 0
        result = []
        for train in self.training_log:
            summ += sum(train) / len(train)
            count += 1
            if count == 10:
                result.append(1-summ/count)
                count = 0
                summ = 0

        df = pd.DataFrame(result)
        df.to_csv("training_error_rate.csv")
        print("Output done.")
        plt.plot(result)
        plt.title("Training Error Rate")
        plt.show()
        plt.savefig('training_error_rate.png')
        plt.clf()

        count = 0
        summ = 0
        result = []
        for test in self.testing_log:
            summ += sum(test) / len(test)
            count += 1
            if count == 10:
                result.append(1-summ/count)
                count = 0
                summ = 0

        df = pd.DataFrame(result)
        df.to_csv("testing_error_rate.csv")
        print("Output done.")
        plt.plot(result)
        plt.title("Testing Error Rate")
        plt.show()
        plt.savefig('testing_error_rate.png')
        plt.clf()

        df = pd.DataFrame(self.loss_log)
        df.to_csv("loss_log.csv")
        print("Output done.")
        plt.plot(self.loss_log)
        plt.title("Training Loss")
        plt.show()
        plt.savefig('loss_log.png')
        plt.clf()

        if Visualize:
            '''
            For visualized network interpretation
            '''
            colors = ['black','red','blue','orange', 'green', 'purple', 'gray', 'cyan', 'olive', 'pink']
            some_list= []
            distro = 0
            for l in self.latent:
                print('\nLatent feature visualization in progress...')
                for f in l:
                    if f[1] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[0], marker='x', label='0' if colors[0] not in some_list else '')
                        if colors[0] not in some_list:
                            some_list.append(colors[0])
                    elif f[1] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[1], marker='x', label='1' if colors[1] not in some_list else '')
                        if colors[1] not in some_list:
                            some_list.append(colors[1])
                    elif f[1] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[2], marker='x', label='2' if colors[2] not in some_list else '')
                        if colors[2] not in some_list:
                            some_list.append(colors[2])
                    elif f[1] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[3], marker='x', label='3' if colors[3] not in some_list else '')
                        if colors[3] not in some_list:
                            some_list.append(colors[3])
                    elif f[1] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[4], marker='x', label='4' if colors[4] not in some_list else '')
                        if colors[4] not in some_list:
                            some_list.append(colors[4])
                    elif f[1] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[5], marker='x', label='5' if colors[5] not in some_list else '')
                        if colors[5] not in some_list:
                            some_list.append(colors[5])
                    elif f[1] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[6], marker='x', label='6' if colors[6] not in some_list else '')
                        if colors[6] not in some_list:
                            some_list.append(colors[6])
                    elif f[1] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[7], marker='x', label='7' if colors[7] not in some_list else '')
                        if colors[7] not in some_list:
                            some_list.append(colors[7])
                    elif f[1] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
                        plt.scatter(f[0][0], f[0][1], c=colors[8], marker='x', label='8' if colors[8] not in some_list else '')
                        if colors[8] not in some_list:
                            some_list.append(colors[8])
                    else:
                        plt.scatter(f[0][0], f[0][1], c=colors[9], marker='x', label='9' if colors[9] not in some_list else '')
                        if colors[9] not in some_list:
                            some_list.append(colors[9])

                plt.legend(loc='upper left')
                some_list.clear()
                plt.show()
                if distro == 0:
                    plt.savefig('Distro0.png')
                    plt.clf()
                    distro += 1
                else: plt.savefig('Distro1.png')
                plt.clf()

        print('\nComplevit Sessionem.')

