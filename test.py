import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import csv

import autoencoder as mn
import cifar_helper as cf
import measure_container as mc

def parse_args():
    desc = "Adversarial Network for Privacy Preserving Communications"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--legit_channel_snr', type=float, default=10,
                        help='SNR of the legitimate channel (dB)')

    parser.add_argument('--adv_channel_snr', type=float, default=5,
                        help='SNR of the adversary\'s channel (dB)')
    return parser.parse_args()

def test_cifar(helper, slug):
    print("\nPlotting random images.")
    image_sample, __ = helper.random_batch(batch_size=9)
    helper.save_images(image_sample, slug)
    return

def run_main(ARGS):

    #####################
    # LOADING THE DATASET
    #####################

    print("\nLoading the dataset.")

    path = "data/CIFAR-10/"
    helper = cf.cifar_helper(path)

    # initialize seed for rng
    seed_rng = 1184445
    np.random.seed(seed_rng)
    tf.set_random_seed(seed_rng)

    ########################
    # SETTING THE PARAMETERS
    ########################

    print("\nSetting the parameters.")

    learning_rate = 1e-4
    batch_train = 32

    num_complete_train = 5

    test_iterations = 10000

    alpha = 0.95

    SNR_legit_dB_train = 10

    SNR_legit_dB_test = ARGS.legit_channel_snr

    SNR_adv_dB_train = 5

    SNR_adv_dB_test = ARGS.adv_channel_snr

    slug_train = "_alpha{0:.2}_A{1}_B{2}".format(alpha, int(SNR_legit_dB_train), int(SNR_adv_dB_train))
    slug_test = "_alpha{0:.2}_A{1}_B{2}".format(alpha, int(SNR_legit_dB_test), int(SNR_adv_dB_test))

    mse_oom = 0.1
    cross_entropy_oom = 2.5
    beta = mse_oom/cross_entropy_oom

    print("Normalization value: {0:.3}".format(beta))

    ###################
    # DEFINING CHANNELS
    ###################

    print("\nDefining standard deviation of the channels.")

    def SNR_to_stddev(SNR_dB):
        stddev = math.sqrt(10**(-SNR_dB/10))
        return stddev

    sigma_legit = SNR_to_stddev(SNR_legit_dB_test)
    sigma_adv = SNR_to_stddev(SNR_adv_dB_test)

    from cifar_helper import img_size, num_channels, num_classes

    image = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='image')
    one_hot_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='one_hot_true')
    cls_true = tf.argmax(one_hot_true, axis=1)

    def legitimate_network():
        with tf.variable_scope('legitimate', reuse=tf.AUTO_REUSE):
            x = mn.encoder(image)
            y = mn.channel(x,sigma_legit)
            image_dec = mn.decoder(y)
        return x, image_dec

    def adversary_network(x):
        with tf.variable_scope('adversary', reuse=tf.AUTO_REUSE):
            z = mn.channel(x,sigma_adv)
            image_adv = mn.decoder(z)
            soft = mn.soft_predictor(image_adv, num_classes=num_classes)
            cls_pred = tf.argmax(soft, axis=1)
        return soft, cls_pred

    x, image_dec = legitimate_network()
    soft, cls_pred = adversary_network(x)

    #################################
    # DEFINING METRICS AND OPTIMIZERS
    #################################

    global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

    def mse_optimizer(t_in, t_out, var_list, learning_rate=1e-3):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        mse = tf.losses.mean_squared_error(t_in, t_out)
        optimizer = optimizer.minimize(mse, var_list=var_list)
        return optimizer, mse

    def cross_entropy_optimizer(t_true, t_pred, var_list, learning_rate=1e-3):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_pred,labels=t_true)
        optimizer = optimizer.minimize(cross_entropy,var_list=var_list)
        return optimizer, cross_entropy

    def mse_cross_entropy_optimizer(t_in, t_out, t_true, t_pred, var_list, alpha=0.2, beta=0.04, learning_rate=1e-3):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        mse = tf.losses.mean_squared_error(t_in, t_out)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_pred,labels=t_true)
        loss = (1-alpha)*mse - beta*alpha*cross_entropy
        optimizer = optimizer.minimize(loss, var_list=var_list)
        return optimizer

    legitimate_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'legitimate')
    adversary_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'adversary')

    _, mse = mse_optimizer(image, image_dec, var_list=legitimate_vars, learning_rate=learning_rate)
    _, cross_entropy = cross_entropy_optimizer(one_hot_true, soft, var_list=adversary_vars, learning_rate=learning_rate)

    #avg_cross_entropy = tf.metrics.mean(cross_entropy)
    correct_prediction = tf.equal(cls_pred, cls_true)
    adversary_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    images_test = helper.images_test
    one_hot_true_test = helper.labels_test

    feed_dict_test = {image: images_test,
                               one_hot_true: one_hot_true_test}

    def test_mse_legit(session):
        mse_test = session.run(mse, feed_dict=feed_dict_test)
        return mse_test

    def test_adversary(session):
        y_pred_test = session.run(cls_pred, feed_dict=feed_dict_test)
        y_true_test = session.run(cls_true, feed_dict=feed_dict_test)
        accuracy_test = session.run(adversary_accuracy, feed_dict=feed_dict_test)
        return y_pred_test, y_true_test, accuracy_test

    images_print = images_test[0:9]
    one_hot_true_print = one_hot_true_test[0:9]

    feed_dict_print = {image: images_print,
                               one_hot_true: one_hot_true_print}

    def test_print_legit(session, slug):
        images_dec_print = session.run(image_dec, feed_dict=feed_dict_print)

        #print("Sample of images at the transmitter side.")
        helper.save_images(images_print, slug, type='tx')
        #print("Sample of images at the receiver side.")
        helper.save_images(images_dec_print, slug, type='rc')
        return

    def test_print_adv(session, slug):
        #print("Adversary predictions.")

        cls_true_print = session.run(cls_true, feed_dict=feed_dict_print)
        cls_pred_print = session.run(cls_pred, feed_dict=feed_dict_print)
        helper.save_images(images_print, slug, type='adv_pred', cls_true=cls_true_print, cls_pred=cls_pred_print)
        return

    def plot_confusion_matrix(y_true, y_pred, classes, counter, slug,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(frameon=False)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        directory = 'figures/figures'+slug
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(directory+'/cm_'+str(counter)+'.png', pad_inches=0.1)

        return ax

    mse_list = []
    acc_list = []

    for i in range(num_complete_train):
        # load trained weights
        saver = tf.train.Saver()
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        directory = "trained/model"+slug_train+"/sim_"+str(i)
        saver.restore(session, directory+"/model.ckpt")
        print("Model restored.")

        mse_t = test_mse_legit(session)
        y_pred_t, y_true_t ,acc_t = test_adversary(session)

        plot_confusion_matrix(y_true=y_true_t, y_pred=y_pred_t, classes=helper.get_class_names(), counter=i, slug=slug_test,
                              normalize=True,
                              title=None,
                              cmap=plt.cm.Blues)

        mse_list.append(mse_t)
        acc_list.append(acc_t)

    def save_results(name, res_list, slug):
        filename = name+"_test_results.csv"
        directory="test/test"+slug
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        filename = filename if directory=="" else directory+"/"+filename
        with open(filename, 'w+') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ')
            csv_writer.writerow(res_list)

    save_results("mse", mse_list, slug=slug_test)
    save_results("acc", acc_list, slug=slug_test)

    return


if __name__== "__main__":
    ARGS = parse_args()
    run_main(ARGS)
