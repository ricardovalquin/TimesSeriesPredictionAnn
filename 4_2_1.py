#feed forward network with back propagation using pybrain
#is very slow...
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import math

#loading data from the file
with open('invemar.txt', 'r') as f:
	lines = f.readlines()

data = [float(e.strip()) for e in lines] #this contains all the data in a list
standarized_data = []
normalized_data = []


#	data standarizaion
mean = 0
for d in range(0, 171, 1):
	mean += data[d]

mean = mean/171
variance = 0
acc = 0
for d in range(0, 171, 1):
	acc = pow(data[d] - mean, 2)

variance = math.sqrt(acc / 171)


# print standarized_data
for d in range(0, len(data), 1):
	standarized_data.append((data[d] - mean)/variance )

#	data normalization
maxim = 0.0
minim = 170.0
for d in range(0, len(standarized_data), 1):
	if(standarized_data[d] > maxim):
		maxim = standarized_data[d]

for d in range(0, 171, 1):
	if(standarized_data[d] < minim):
		minim = standarized_data[d]

print "maxim standarized value: %f" % maxim
print "minim standarized value: %f" % minim

d1 = 0
d2= 1
print "normalizando datos..."
for d in range(0, 171, 1):
	normalized_data.append(((standarized_data[d] - minim)* (d2 - d1) / (maxim - minim)) + d1)

print normalized_data

#creating the data set
trainingSet = SupervisedDataSet(4, 1)
for i in range(0, 115, 1):
	trainingSet.addSample((normalized_data[i], normalized_data[i+1], normalized_data[i+2], normalized_data[i+3]), (normalized_data[i+4]))#116 tuples

# print "trainginSet"
# for inpt, target in trainingSet:
# 	print inpt, target

ds = SupervisedDataSet(4, 1)
for i in range(119, 162, 1):
	ds.addSample((normalized_data[i], normalized_data[i+1], normalized_data[i+2], normalized_data[i+3]), (normalized_data[i+4]))#44 tuples

# print "dataset"
# for inpt, target in ds:
# 	print inpt, target

net = buildNetwork(4, 2, 1, bias = True, hiddenclass = SigmoidLayer)
# net = buildNetwork(3, 4, 1, bias = True, hiddenclass = TanhLayer)

trainer = BackpropTrainer(net, ds, learningrate = 0.001, momentum = 0.99)
print "entrenando hasta converger..."
print "training until the Convergence: ", trainer.trainUntilConvergence(verbose=True,
															trainingData=trainingSet,
															validationData=ds,
															maxEpochs=100)

print "predicting the next month: 0.28733060532417387"
y = net.activate([0.12897219396266188, 0.10031611360730924, 0.13198268361816343, 0.239516498322237])
print y

print "des normalizando"
#((y - d1) * (x_max - x_min) / (d2 - d1)) + x_min= x
x = ((y - d1) *(maxim - minim) / (d2 - d1)) + maxim
print x

print "des estandarizando"

dato = (x * variance) + mean

print "dato des estandarizado"
print dato
	