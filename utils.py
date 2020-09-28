import numpy as np
import sys
import inspect

def raiseNotDefined():
  print ("Method not implemented: %s" % inspect.stack()[1][3])
  sys.exit(0)

def GradLeakReLU (x):
    if x > 0: return 1
    else: return 0.01

def Softmax (x):
    e_x = np.exp(x - np.max(x))
    output = (e_x / e_x.sum()).tolist()

    return output

def CrossEntropy (Outputs, Truth):
    Error = 0
    for i in range(len(Outputs)):
        Outputs[i] = max(Outputs[i], 1e-12)
        Dummy = max((1 - Outputs[i], 1e-12))
        Error += Truth[i] * np.log10(Outputs[i]) + (1 - Truth[i]) * np.log10(Dummy)
 
    return -Error

def GradEntropy (Outputs, Truth):
    Grad = []
    for i in range(len(Outputs)):
        Grad.append( -1 * ( Truth[i]/Outputs[i] + (1-Truth[i])/(1-Outputs[i]) ) )

    return Grad

def GradEntropy_Softmax (Outputs, Truth):
    Grad = []
    for i in range(len(Outputs)):
        Grad.append(Outputs[i] - Truth[i])

    return Grad

def get_max(my_list):
    m = None
    for item in my_list:
        if isinstance(item, list):
            item = get_max(item)
        if not m or m < abs(item):
            m = abs(item)

    return m

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>', printEnd = '\r', test = 0, train = 0):
    #█
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s %% %s Test:%.2f Train:%.2f' % (prefix, bar, percent, suffix, test, train), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()