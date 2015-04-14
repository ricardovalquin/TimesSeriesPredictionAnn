from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#creating the data set
ds = SupervisedDataSet(3, 1)

ds.addSample((122.1088395697, 78.3859367463, 41.2508418791), (65.4751234025,))
ds.addSample((78.3859367463, 41.2508418791, 65.4751234025), (81.6517046411,))
ds.addSample((41.2508418791, 65.4751234025, 81.6517046411), (96.7179955411,))
ds.addSample((65.4751234025, 81.6517046411, 96.7179955411), (128.1517557891,))
ds.addSample((81.6517046411, 96.7179955411, 128.1517557891), (157.4746710799,))
ds.addSample((96.7179955411, 128.1517557891, 157.4746710799), (123.5395109437,))
ds.addSample((128.1517557891, 157.4746710799, 123.5395109437), (153.1482529168,))
ds.addSample((157.4746710799, 123.5395109437, 153.1482529168),(80.3077508874,))
ds.addSample((123.5395109437, 153.1482529168, 80.3077508874),(129.5673782597,))
ds.addSample((153.1482529168, 80.3077508874, 129.5673782597),(89.6068461264,))
ds.addSample((80.3077508874, 129.5673782597, 89.6068461264),(88.3411357835,))
ds.addSample((129.5673782597, 89.6068461264, 88.3411357835),(67.55426552,))
print "len of ds: ", len(ds)

net = buildNetwork(3, 4, 1, bias = True, hiddenclass = SigmoidLayer)
# net = buildNetwork(3, 4, 1, bias = True, hiddenclass = TanhLayer)

#creating the network and the layers
# net = FeedForwardNetwork()
# inLayer = LinearLayer(3)
# hiddenLayer = SigmoidLayer(4)
# outLayer = LinearLayer(1)

#adding the layers to the network
#net.addInputModule(inLayer)
#net.addInputModule(hiddenLayer)
#net.addInputModule(outLayer)

#creating all the conections
#in_to_hidden = FullConnection(inLayer, hiddenLayer)
#hidden_to_out = FullConnection(hiddenLayer, outLayer)
#adding the conections to the network
#net.addConnection(in_to_hidden)
#net.addConnection(hidden_to_out)

#net.sortModules()


trainer = BackpropTrainer(net, ds)
#print "train only once: ", trainer.train()

print "training until the Convergence: ", trainer.trainUntilConvergence()

print "predicting the next month: "
print net.activate([110.0274337672, 112.7894792365, 97.7697551271])
