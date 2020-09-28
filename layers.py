import utils
import numpy as np

class Output:

    def __init__(self):
        self.Outputs = []
        self.crossentropy_log = []
        self.confusion = []
        for i in range(10):
            temp = []
            for j in range(10):
                temp.append(0)
            self.confusion.append(temp)

        self.right = 0
        self.log = []
        self.feature = []
        self.Grad_pass = []

    def update(self, Inputs, Truth, Feature, fin):
        self.Inputs = Inputs[:]

        '''
        Latent Feature
        '''
        if Feature == True:
            self.feature.append([Inputs[:], Truth])
        '''
        Latent Feature
        '''
        self.Outputs.clear()
        self.Outputs = utils.Softmax(self.Inputs)

        '''
        prediction
        '''
        pred = 0
        gros = self.Outputs[0]
        for i in range(len(self.Outputs)):
            if gros < self.Outputs[i]:
                gros = self.Outputs[i]
                pred = i

        if fin == True:
            for i in range (len(Truth)):
                if Truth[i] == 1: truth = i
            self.confusion[truth][pred] += 1

        if Truth[pred] == 1: 
            self.log.append(1)
            self.right = 1
        else: 
            self.log.append(0)
            self.right = 0

        self.crossentropy_log.append(utils.CrossEntropy(self.Outputs, Truth))
        self.Grad_pass.append(utils.GradEntropy_Softmax(self.Outputs, Truth))

class Hidden:

    def __init__(self, n_nodes, n_outputs, RandomInit):
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.W = []
        self.b = 0.1
        self.B = np.full(n_outputs, 0.1)
        self.Outputs = np.zeros(n_nodes)
        row = []
        for i in range(self.n_nodes):
            if RandomInit == False:
                row.clear()
                for j in range(self.n_outputs):
                    row.append(0.0)
            else:
                row.clear()
                for j in range(self.n_outputs):
                    row.append(0.5-np.random.random_sample())
            
            self.W.append(row[:])

        self.W = np.array(self.W)
        self.W.transpose()

    def activate_pass(self, Inputs):
        Inputs = np.array(Inputs)
        self.Outputs = np.where(Inputs > 0, Inputs, 0.01*Inputs)
        self.Z = np.dot(self.Outputs, self.W) + self.B

    def update(self, Error_e, LearningRate):
        self.learningrate = LearningRate
        self.Grad_pass = []
        
        e = np.array(Error_e)
        if np.sum(np.abs(e)) > e.size*10:
            e /= np.sum(np.abs(e)) / (e.size * 10)
        self.Grad_error = np.dot(self.W, e)
        temp = np.zeros((e.shape[0], self.Outputs.shape[0]))
        for i in range(e.shape[0]):
            temp[i] = self.Outputs * e[i]
        self.W -= np.multiply(self.learningrate, temp.T)

        for i in range(self.n_nodes):
            self.Grad_pass.append(self.Grad_error[i] * utils.GradLeakReLU(self.Outputs[i]))

    def regularize(self):
        if np.sum(np.abs(self.W)) > self.W.size*10:
            self.W /= np.sum(np.abs(self.W)) / (self.W.size * 10)
        
        
        
