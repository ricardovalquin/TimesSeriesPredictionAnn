#feed forward network with back propagation using pybrain
#is very slow...
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#loading data from the file
with open('invemar.txt', 'r') as f:
	lines = f.readlines()

data = [float(e.strip()) for e in lines] #this contains all the data in a list

#creating the data set

trainingSet = SupervisedDataSet(3, 1)
for i in range(0, 116, 1):
	trainingSet.addSample((data[i], data[i+1], data[i+2]), (data[i+3]))#116 tuples

print "trainginSet"
for inpt, target in trainingSet:
	print inpt, target

ds = SupervisedDataSet(3, 1)
for i in range(119, 163, 1):
	ds.addSample((data[i], data[i+1], data[i+2]), (data[i+3]))#44 tuples

print "dataset"
for inpt, target in ds:
	print inpt, target


net = buildNetwork(3, 4, 1, bias = True, hiddenclass = SigmoidLayer)
# net = buildNetwork(3, 4, 1, bias = True, hiddenclass = TanhLayer)

trainer = BackpropTrainer(net, ds, learningrate = 0.001, momentum = 0.99)

print "training until the Convergence: ", trainer.trainUntilConvergence(verbose=True,
                              								trainingData=trainingSet,
								                            validationData=ds,
								                            maxEpochs=100)

print "predicting the next month: 85.7849732991"
print net.activate([56.7991019814, 61.7071875259, 78.3741382349])
print "predicting the next month: 59.5832360614"
print net.activate([61.7071875259, 78.3741382349, 85.7849732991])
