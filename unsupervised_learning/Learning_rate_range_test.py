# standard libraries 
import math
# tensorflow & keras
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
import keras.backend as K
# plots 
from matplotlib import pyplot as plt
# custom made imports 
from model1 import ConvAE_clustering

"""
Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
See for details: https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

@author: Sterre de Jonge (2022)
"""

class LRFinder:
    
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.val_losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        print(logs)
        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 50:
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, start_lr, end_lr, batch_size=64, epochs=8):
        self.epochs=epochs
        num_batches = epochs * x_train.samples / batch_size
        self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.H = self.model.fit(x= x_train, 
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[callback], 
                        verbose=1)

                        # TODO: nu wordt voor elke batch de loss etc. geprint, dat is irritant. 

        # Restore the weights to the state before model fitting
        self.model.load_weights('tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)

if __name__ == "__main__":

    # set paramaters
    save_directory = '/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/'
    directory_data = '/Users/Sterre/files/PROCESSED/mels_7/Training'
    dim = (128, 128)
    batch_size = 64 # 64 / 256
    epochs = 6
    loss = 'mse'
    n_features = 32
    time_stamp = '11122022'
    optimizer = 'SGD' # of ADAM / SGD

    datagen = ImageDataGenerator(rescale=1./255) 
    train_data = datagen.flow_from_directory(
                    directory = directory_data, 
                    target_size = dim, 
                    batch_size = batch_size,
                    class_mode = 'input', 
                    color_mode = 'grayscale', 
                    shuffle=True
            )

    autoencoder = ConvAE_clustering.CAE(input_shape=(128, 128, 1), kernel_size=[5,5,3], 

                                        activation = 'relu', filters=[32, 64, 128], 
                                        features=n_features, batch_norm=True)
    if optimizer == 'ADAM': opt = keras.optimizers.Adam()
    elif optimizer == 'SGD': opt = keras.optimizers.SGD()
    autoencoder.compile(optimizer=opt, loss=loss) 

    lr_finder = LRFinder(autoencoder)
    lr_finder.find(train_data, start_lr=0.000000001, end_lr=100, batch_size=batch_size, epochs=epochs)
    lr_finder.plot_loss(n_skip_beginning=1, n_skip_end=3) # to inspect the loss-learning rate graph
    print("learning rate range test has been performed")
    plt.savefig(save_directory + '/learning_rate_range_test_{}_{}_{}.eps'.format(optimizer, batch_size, time_stamp), format='eps')