import os
import sys
cifar_path = 'cifar_utils'
sys.path.insert(0, cifar_path)

import numpy as np
import matplotlib.pyplot as plt
import cifar10

default_path = "data/CIFAR-10"

# Data dimensions
from cifar10 import img_size, num_channels, num_classes

class cifar_helper:

    def __init__(self, path=default_path):
        cifar10.data_path = path
        cifar10.maybe_download_and_extract()
        self.class_names = cifar10.load_class_names()
        self.images_train, self.cls_train, self.labels_train = cifar10.load_training_data()
        self.images_test, self.cls_test, self.labels_test = cifar10.load_test_data()
        self.counter = 0
        return

    # Helper function for plotting images
    def save_images(self, images, slug="", type='example', cls_true=None, cls_pred=None, smooth=True, mode='train'):

        assert len(images) == 9

        # Create figure with sub-plots.
        fig, axes = plt.subplots(3, 3)

        # Adjust vertical spacing if we need to print ensemble and best-net.
        if cls_pred is None:
            hspace = 0.3
        else:
            hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            # Plot image.
            ax.imshow(images[i, :, :, :],
                      interpolation=interpolation)

            # Show true and predicted classes.
            if cls_true is None:
                xlabel = ""
            else:
                # Name of the true class.
                cls_true_name = self.class_names[cls_true[i]]
                if cls_pred is None:
                    xlabel = "True: {0}".format(cls_true_name)
                else:
                    # Name of the predicted class.
                    cls_pred_name = self.class_names[cls_pred[i]]

                    xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        #plt.show(block = False)

        directory = 'results/figures'+slug
        if mode == 'train':
            directory = 'results/train/figures'+slug
        if mode == 'test':
            directory = 'results/test/figures'+slug

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(directory+'/'+str(self.counter)+'_'+type+'.png' ,dpi = 200)
        self.counter = self.counter + 1

        return

    def random_batch(self, batch_size):
        # Number of images in the training-set.
        num_images = len(self.images_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=batch_size,
                               replace=False)

        # Use the random index to select random images and labels.
        x_batch = self.images_train[idx, :, :, :]
        y_batch = self.labels_train[idx, :]

        return x_batch, y_batch

    def get_class_names(self):
        return self.class_names
