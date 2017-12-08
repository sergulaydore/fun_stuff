'''
author: Sergul Aydore
References:
    keras: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
    gluon: http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-dropout-gluon.html
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import nd
import time
import matplotlib.pyplot as plt

class GluonvsKerasMNIST(object):
    def __init__(self, batch_size=128, epochs=20,
                 num_outputs=10, num_hidden=512,
                 dropout_prop=0.5, learning_rate=0.01):
        # get data
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train / 255
        x_test = x_test / 255
        self.x_train = x_train.reshape(x_train.shape[0],
                                       x_train.shape[1] ** 2)
        self.x_test = x_test.reshape(x_test.shape[0],
                                     x_test.shape[1] ** 2)
        # set parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.dropout_prop = dropout_prop
        self.lr = learning_rate

    def keras_model(self):
        y_train = keras.utils.to_categorical(self.y_train, self.num_outputs)
        y_test = keras.utils.to_categorical(self.y_test, self.num_outputs)
        sgd = keras.optimizers.SGD(lr=self.lr, momentum=0.0,
                                   decay=0.0, nesterov=False)
        model = Sequential()
        model.add(Dense(self.num_hidden, activation='relu',
                        input_shape=(self.x_train.shape[1],)))
        model.add(Dropout(self.dropout_prop))
        model.add(Dense(self.num_hidden, activation='relu'))
        model.add(Dropout(self.dropout_prop))
        model.add(Dense(self.num_outputs, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        results = model.fit(self.x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(self.x_test, y_test))
        return results

    def gluon_model(self):
        ctx = mx.cpu()

        data_iter_train = mx.gluon.data.DataLoader(
            mx.gluon.data.ArrayDataset(data=self.x_train,
                                       label=self.y_train),
            batch_size=self.batch_size)
        data_iter_test = mx.gluon.data.DataLoader(
            mx.gluon.data.ArrayDataset(data=self.x_test,
                                       label=self.y_test),
            batch_size=self.batch_size)

        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(self.num_hidden, activation="relu"))
            net.add(gluon.nn.Dropout(self.dropout_prop))
            net.add(gluon.nn.Dense(self.num_hidden, activation="relu"))
            net.add(gluon.nn.Dropout(self.dropout_prop))
            net.add(gluon.nn.Dense(self.num_outputs))

        net.collect_params().initialize(ctx=ctx)
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd',
                                {'learning_rate': self.lr})

        def evaluate_accuracy(data_iterator, net):
            acc = mx.metric.Accuracy()
            for batch in data_iterator:
                data = batch[0]
                data = data.as_in_context(ctx)
                label = batch[1]
                label = label.as_in_context(ctx)
                output = net(data)
                predictions = nd.argmax(output, axis=1)
                acc.update(preds=predictions, labels=label)
            return acc.get()[1]

        test_accuracy = []
        train_accuracy = []

        for e in range(self.epochs):
            print(e)
            for i, batch in enumerate(data_iter_train):
                data = batch[0].as_in_context(ctx)
                label = batch[1].as_in_context(ctx)
                with autograd.record():
                    output = net(data)
                    loss = softmax_cross_entropy(output, label)
                    loss.backward()
                trainer.step(data.shape[0])

            test_accuracy.append(evaluate_accuracy(data_iter_test, net))
            train_accuracy.append(evaluate_accuracy(data_iter_train, net))

        results = {'acc': train_accuracy,
                   'val_acc': test_accuracy}
        return results

if __name__ == '__main__':
    comparison = GluonvsKerasMNIST()

    tick_keras = time.time()
    results_keras = comparison.keras_model()
    time_keras = time.time() - tick_keras

    tick_gluon = time.time()
    results_gluon = comparison.gluon_model()
    time_gluon = time.time() - tick_gluon

    print("Took %.2f seconds for Keras and %.2f seconds for Gluon"
          %(time_keras, time_gluon))

    plt.figure()
    plt.plot(results_keras.history['val_acc'], 'b.-.' ,label = "Keras - validation")
    plt.plot(results_keras.history['acc'], 'b-' ,label = "Keras - training")
    plt.plot(results_gluon['val_acc'], 'r.-.' ,label = "Gluon - validation")
    plt.plot(results_gluon['acc'], 'r-' ,label = "Gluon - training")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title("MNIST classification for 512-512 network with dropout 0.5")
    plt.legend()
    plt.ylim(0.90, .96)
    plt.show()