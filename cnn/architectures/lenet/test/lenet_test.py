from cnn.architectures.lenet.lenet import LeNet
from utils.dataset_loader import DatasetLoader

dl = DatasetLoader()
dl.load_mnist_data()
(trainX, trainY), (testX, testY) = dl.mnist_data

lenet = LeNet(dl.mnist_data)
#lenet.run_training()
lenet.evaluate(5)