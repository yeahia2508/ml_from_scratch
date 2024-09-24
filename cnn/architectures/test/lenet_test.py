from cnn.architectures.lenet import *
from utils.dataset_loader import *

dl = DatasetLoader()
dl.load_mnist_data()
(trainX, trainY), (testX, testY) = dl.mnist_data

lenet = LeNet(dl.mnist_data)
lenet.run_training()