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

    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Quality-security tradeoff value')

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

def pmf_equalizer_v2(cluster_vector):
    num_input = len(cluster_vector)
    E = np.zeros((num_input,num_input))
    num_clusters = len(np.unique(cluster_vector))
    for j in range(num_input):
        for i in range(num_input):
            if (cluster_vector[i] == cluster_vector[j]):
                E[i,j]=1
    E = E/np.sum(E, axis=1)
    return E, num_clusters

def get_confusion_matrix(y_true, y_pred, normalize=True):
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm / cm.astype(np.float).sum(axis=1)
    return cm, title

def plot_confusion_matrix_v2(cm, classes, counter, directory,
                          title=None,
                          cmap=plt.cm.Blues):

        if not title:
            title = 'Confusion matrix'

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

        #directory = 'figures/figures'+slug
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(directory+'/cm_'+str(counter)+'.png', pad_inches=0.0, bbox_inches = 'tight', dpi = 200)

        return ax

def run_main(ARGS):

    #####################
    # LOADING THE DATASET
    #####################

    print("\nLoading the dataset.")

    path = "data/CIFAR-10/"
    helper = cf.cifar_helper(path)

    # initialize seed for rng
    seed_rng = 1184445
    #np.random.seed(seed_rng)
    #tf.set_random_seed(seed_rng)

    ########################
    # SETTING THE PARAMETERS
    ########################

    print("\nSetting the parameters.")

    learning_rate = 1e-4
    batch_train = 128

    num_complete_train = 1

    train_epochs = 4

    prelim_iterations = 3
    legitimate_iterations = 5
    adversary_iterations = 2

    test_iterations = 10

    alpha = ARGS.alpha

    SNR_legit_dB = ARGS.legit_channel_snr
    SNR_adv_dB = ARGS.adv_channel_snr

    # cluster vector: classes with same value are put in the same cluster
    # classes are: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
    # 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    cluster_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    slug = "_alpha{0:.2}_M{1}_E{2}".format(alpha, int(SNR_legit_dB), int(SNR_adv_dB))

    ###################
    # DEFINING CHANNELS
    ###################

    print("\nDefining standard deviation of the channels.")

    def SNR_to_stddev(SNR_dB):
        stddev = 10**(-SNR_dB/20)
        return stddev

    sigma_legit = SNR_to_stddev(SNR_legit_dB)
    sigma_adv = SNR_to_stddev(SNR_adv_dB)

    ##############################
    # DEFINING TENSOR PLACEHOLDERS
    ##############################

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

    # defining equalized tensor
    E, num_clusters = pmf_equalizer_v2(cluster_vector)
    equalized = tf.matmul(tf.cast(one_hot_true, tf.float64), tf.convert_to_tensor(E))

    slug = slug + "_k{0}".format(num_clusters)

    test_cifar(helper, slug)

    # plotting equalization matrix
    eq_directory = "results/figures/eq_v2_clusters{0}".format(num_clusters)
    eq_title = "Equalization matrix with k={0} clusters".format(num_clusters)
    plot_confusion_matrix_v2(E, classes=helper.get_class_names(), counter=0, directory=eq_directory,
                          title=eq_title,
                          cmap=plt.cm.Blues)

    #################################
    # DEFINING METRICS AND OPTIMIZERS
    #################################

    global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

    def mse_optimizer(t_in, t_out, var_list, learning_rate=1e-4):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        mse = tf.losses.mean_squared_error(t_in, t_out)
        optimizer = optimizer.minimize(mse, var_list=var_list)
        return optimizer, mse

    def cross_entropy_optimizer(t_true, t_pred, var_list, learning_rate=1e-4):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_pred,labels=t_true)
        optimizer = optimizer.minimize(cross_entropy,var_list=var_list)
        return optimizer, cross_entropy

    def mse_cross_entropy_optimizer(t_in, t_out, t_false, t_pred, var_list, alpha=1.0, learning_rate=1e-4):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        mse = tf.losses.mean_squared_error(t_in, t_out)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_pred,labels=t_false)
        loss = mse + alpha*cross_entropy
        optimizer = optimizer.minimize(loss, var_list=var_list)
        return optimizer

    legitimate_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'legitimate')
    adversary_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'adversary')

    optimizer1, mse = mse_optimizer(image, image_dec, var_list=legitimate_vars, learning_rate=learning_rate)
    optimizer2, cross_entropy = cross_entropy_optimizer(one_hot_true, soft, var_list=adversary_vars, learning_rate=learning_rate)
    optimizer3 = mse_cross_entropy_optimizer(image, image_dec, equalized, soft, var_list=legitimate_vars, alpha=alpha, learning_rate=learning_rate)

    #avg_cross_entropy = tf.metrics.mean(cross_entropy)
    correct_prediction = tf.equal(cls_pred, cls_true)
    adversary_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #############################
    # DEFINING TRAINING FUNCTIONS
    #############################

    def train_legit_prelim(session, num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            image_batch, one_hot_true_batch = helper.random_batch(batch_size=batch_train)

            feed_dict_train = {image: image_batch,
                               one_hot_true: one_hot_true_batch}

            i_global, _ = session.run([global_step, optimizer1],
                                      feed_dict=feed_dict_train)

        end_time = time.time()
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return

    def train_legit_minimax(session, num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            image_batch, one_hot_true_batch = helper.random_batch(batch_size=batch_train)

            feed_dict_train = {image: image_batch,
                               one_hot_true: one_hot_true_batch}

            i_global, _ = session.run([global_step, optimizer3],
                                      feed_dict=feed_dict_train)

        end_time = time.time()
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return

    def train_adversary(session, num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            image_batch, one_hot_true_batch = helper.random_batch(batch_size=batch_train)

            feed_dict_train = {image: image_batch,
                               one_hot_true: one_hot_true_batch}

            i_global, _ = session.run([global_step, optimizer2],
                                      feed_dict=feed_dict_train)

        end_time = time.time()
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return

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
        cross_entropy_test = np.mean(session.run(cross_entropy, feed_dict=feed_dict_test))
        return accuracy_test, y_pred_test, y_true_test, cross_entropy_test

    # DEFINING TEST FUNCTIONS

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

    #######################
    # INITIALIZE CONTAINERS
    #######################

    mse_container = mc.measure_container(aim="low")
    acc_container = mc.measure_container(aim="high", MAX_VALUE=100)

    directory="C:/Users/Thomas/Dropbox/PPCAN/"+"results/trainv2/recadv"
    mse_filename = "mse"+slug+".csv"
    acc_filename = "acc"+slug+".csv"

    ##########################
    # DEFINING TRAINING EPOCHS
    ##########################

    def run_train_epoch():
        #training of the legitimate channel
        num_iterations = legitimate_iterations
        mse_container.update_iter_list(num_iterations)
        print("Training legitimate channel for {0} iterations.".format(num_iterations))
        train_legit_minimax(session,num_iterations)
        mse_t = test_mse_legit(session)
        legit_total_iterations = mse_container.get_current_iter()
        print("MSE of the legitimate channel after {0} total iterations: {1:.2}".format(legit_total_iterations, mse_t))
        accuracy_t, _, _, _ = test_adversary(session)
        #print("Cross entropy of adversary's softmax outputs: {0:.4}".format(cross_entropy_t))
        print("Accuracy of adversary's predictions: {0:.4}".format(accuracy_t))
        mse_container.append_elem(mse_t)
        print("")

        #training of the advarsary
        num_iterations=adversary_iterations
        acc_container.update_iter_list(num_iterations)
        print("Training adversary for {0} iterations.".format(num_iterations))
        train_adversary(session,num_iterations)
        accuracy_t, _, _, cross_entropy_t = test_adversary(session)
        print("Cross entropy of adversary's softmax outputs after {0} total iterations: {1:.4}".format(adv_total_iterations, cross_entropy_t))
        print("Accuracy of adversary's predictions after {0} total iterations: {1:.4}".format(adv_total_iterations, accuracy_t))
        #mse_0 = test_mse_legit(session)
        #print("MSE of the legitimate channel (should remain the same): {0:.2}".format(mse_0))
        print("")
        acc_container.append_elem(accuracy_t)

        return

    for i in range(num_complete_train):
        sim_start_time = time.time()
        print("")
        print("----------------------")
        print("COMPLETE TRAINING: {0}".format(i+1))
        print("----------------------")

        mse_container.initialize_list()
        mse_container.initialize_iter_list()
        acc_container.initialize_list()
        acc_container.initialize_iter_list()

        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        legit_total_iterations = 0
        adv_total_iterations = 0
        mse_0 = test_mse_legit(session)

        print("MSE of the legitimate channel before any iteration: {0:.2}".format(mse_0))

        accuracy_0, _, _, cross_entropy_0 = test_adversary(session)

        print("Cross_entropy of adversary's softmax outputs before any iteration: {0:.4}".format(cross_entropy_0))
        print("Accuracy of adversary's predictions  before any iteration: {0:.4}".format(accuracy_0))

        print("\nPreliminary training.")

        # PRELIMINARY TRAINING OF THE LEGITIMATE CHANNEL
        train_legit_prelim(session,prelim_iterations)
        mse_container.update_iter_list(0)
        mse_1 = test_mse_legit(session)
        mse_container.append_elem(mse_1)
        print("MSE of the legitimate channel after preliminary training: {0:.2}".format(mse_1))

        print("")

        # PRELIMINARY TRAINING OF THE ADVERSARY'S NETWORK
        train_adversary(session,prelim_iterations)
        accuracy_1, _, _, cross_entropy_1 = test_adversary(session)
        acc_container.update_iter_list(0)
        acc_container.append_elem(accuracy_1)
        print("Cross_entropy of adversary's softmax outputs after preliminary training: {0:.4}".format(cross_entropy_1))
        print("Accuracy of adversary's predictions after preliminary training: {0:.4}".format(accuracy_1))

        for j in range(train_epochs):
            print("Train epoch: {0}\n".format(j+1))
            run_train_epoch()

        test_print_legit(session, slug+"_recadv")
        test_print_adv(session, slug+"_recadv")

        mse_container.append_list()
        acc_container.append_list()

        mse_container.write_matrix(mse_filename, directory)
        acc_container.write_matrix(acc_filename, directory)

        _, y_pred_t, y_true_t, _ = test_adversary(session)
        cm, cm_title = get_confusion_matrix(y_true_t, y_pred_t, normalize=True)
        cm_directory = directory+"/figures"+slug
        plot_confusion_matrix_v2(cm, classes=helper.get_class_names(), counter=i, directory=cm_directory, title=cm_title)

        saver = tf.train.Saver()
        save_dir = "trained/model"+slug+"/sim_"+str(i)
        if save_dir != "" and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = saver.save(session, save_dir+"/model.ckpt")
        print("Model saved in path: %s" % save_path)

        sim_end_time = time.time()
        delta_time = sim_end_time - sim_start_time

        # Print the time-usage.
        print("Overall time elapsed: " + str(timedelta(seconds=int(round(delta_time)))))
    return


if __name__== "__main__":
    ARGS = parse_args()
    run_main(ARGS)
