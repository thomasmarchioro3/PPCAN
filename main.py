import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import os
import sys
modules_path = "modules"
sys.path.insert(0, modules_path)

import cifar10
import args_parser
from model import get_model

DATASETS = {"cifar": cifar10}

def run_main(ARGS):
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(ARGS).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")

    bandwidth = ARGS.conv_depth/(16*3)
    print("\nBandwidth: ", bandwidth)

    dirname="mode{mode}/alpha{alpha:.2f}B{b}E{e}band{band:.2f}".format(
        mode = ARGS.loss_type,
        alpha = ARGS.alpha,
        b = int(ARGS.snr_legit_train),
        e = int(ARGS.snr_adv_train),
        band = bandwidth
    )

    train_dir = ARGS.train_dir + "/" + dirname + "/train"
    save_dir = ARGS.train_dir + "/" + dirname + "/saved"
    test_dir = ARGS.test_dir + "/" + dirname + "/test"
    for dir in [train_dir, save_dir, test_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    img_height = DATASETS[ARGS.dataset]._HEIGHT
    img_width = DATASETS[ARGS.dataset]._WIDTH
    num_channels = DATASETS[ARGS.dataset]._NUM_CHANNELS
    num_classes = DATASETS[ARGS.dataset]._NUM_CLASSES

    u = tf.compat.v1.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name='u')
    p = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='p')

    snr_legit = ARGS.snr_legit_train
    snr_adv = ARGS.snr_adv_train

    model_vars, model_metrics, model_losses, model_collections = get_model(u, p,
                                                    params={ 'ARGS': ARGS,
                                                             'snr_legit': snr_legit,
                                                             'snr_adv': snr_adv})

    DATASETS[ARGS.dataset].data_path = ARGS.data_dir_test
    DATASETS[ARGS.dataset].maybe_download_and_extract()
    u_test, s_test, p_test = DATASETS[ARGS.dataset].load_test_data()

    def test(session):
        feed_dict_test = { u: u_test,
                           p: p_test
        }
        results = session.run(model_metrics,
                feed_dict=feed_dict_test)
        return results

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    if ARGS.mode in ("train", "Train"):
        DATASETS[ARGS.dataset].data_path = ARGS.data_dir_train
        DATASETS[ARGS.dataset].maybe_download_and_extract()
        u_train, s_train, p_train = DATASETS[ARGS.dataset].load_training_data()
        batch_size = ARGS.batch_size
        train_iters_legit_1 = ARGS.train_iters_legit_1
        train_iters_legit_3 = ARGS.train_iters_legit_3
        train_iters_adv_2 = ARGS.train_iters_adv_2
        train_iters_adv_3 = ARGS.train_iters_adv_3

        global_step = tf.Variable(initial_value=0,
                              name='global_step', trainable=False)

        legit_vars = model_collections['legitimate_vars']
        adv_vars = model_collections['adversary_vars']

        optimizers = [
            tf.train.AdamOptimizer(learning_rate=ARGS.learn_rate).minimize(model_losses['loss_legit_prelim'], var_list=legit_vars),
            tf.train.AdamOptimizer(learning_rate=ARGS.learn_rate).minimize(model_losses['loss_adv'], var_list=adv_vars),
            tf.train.AdamOptimizer(learning_rate=ARGS.learn_rate).minimize(model_losses['loss_legit'], var_list=legit_vars)
        ]

        def get_random_batch(batch_size):
            idx = np.random.choice(len(u_train),
                                   size=batch_size)
            u_batch = u_train[idx, :, :, :]
            p_batch = p_train[idx, :]
            return u_batch, p_batch

        def train(session, optimizer, iters):
            for i in range(iters):
                u_batch, p_batch = get_random_batch(batch_size)
                feed_dict_train = {u: u_batch,
                                   p: p_batch}
                i_global, _ = session.run([global_step, optimizer],
                                          feed_dict=feed_dict_train)

        for n in range(ARGS.num_simulations):
            print("Beginning simulation number: ", n)
            sim_slug = "/sim{}".format(n)
            for dir in [train_dir, save_dir, test_dir]:
                if not os.path.exists(dir+sim_slug):
                    os.makedirs(dir+sim_slug)

            session = tf.Session(config = session_config)
            init = tf.compat.v1.global_variables_initializer()
            session.run(init)

            for phase in range(3):
                measures = {
                    'legit_iters': [],
                    'adv_iters': [],
                    'mse': [],
                    'psnr': [],
                    'cross_entropy': [],
                    'accuracy': [],
                    'avg_power_y': [],
                    'avg_power_z': []
                }

                def write_results():
                    filename = train_dir+sim_slug+"/results_"+str(phase+1)+".csv"

                    df = pd.DataFrame(measures)
                    df.to_csv(filename)
                    print("\tResults saved in ", filename)


                t_epoch = 0
                tot_iters_legit = 0
                tot_iters_adv = 0
                test_results = test(session)

                def append_results(results):
                    print("\t -Total main network iterations: ", tot_iters_legit)
                    print("\t -Total adversary iterations: ", tot_iters_adv)
                    measures['legit_iters'].append(tot_iters_legit)
                    measures['adv_iters'].append(tot_iters_adv)
                    for key in results:
                        if key in measures:
                            if t_epoch % 1 == 0:
                                print("\t -", key, ": {0}".format(results[key]))
                            measures[key].append(results[key])

                if phase == 0:
                    # Phase 1: preliminary training of main network
                    print("\tPhase 1: Preliminary training of main network.")
                    saver = tf.compat.v1.train.Saver(var_list=legit_vars)
                    save_path = save_dir + sim_slug + "/model_phase_1.ckpt"
                    if os.path.exists(save_path+".index") and not ARGS.delete_prev_model_1:
                        saver.restore(session, save_path)
                        print("\tModel restored from: ", save_path)
                        if ARGS.skip_phase_1:
                            print("\tSkipping phase 1")
                            continue

                    start_time = time.time()
                    test_results = test(session)
                    append_results(test_results)
                    mse_new = test_results['mse']

                    while True:
                        t_epoch = t_epoch + 1
                        print("\tEpoch number: ", t_epoch)
                        mse_old = mse_new
                        train(session, optimizers[0], train_iters_legit_1)
                        test_results = test(session)
                        tot_iters_legit = tot_iters_legit + train_iters_legit_1
                        append_results(test_results)
                        write_results()
                        mse_new = test_results['mse']
                        # save model every 10 epochs
                        if t_epoch % 10 == 0:
                            saver.save(session, save_path)
                            print("\tModel saved in path: ", save_path)
                        # if a specific condition is satisfied, exit from the loop
                        condition = t_epoch >= ARGS.train_epochs_1
                        if ARGS.reach_convergence_1:
                            condition = abs(mse_old - mse_new) < ARGS.mse_epsilon_1
                        if condition:
                            break

                elif phase == 1:
                    # Phase 2: preliminary training of the adversary
                    print("\tPhase 2: Preliminary training of the adversary.")
                    saver = tf.compat.v1.train.Saver(var_list=adv_vars)
                    save_path = save_dir + sim_slug + "/model_phase_2.ckpt"
                    if os.path.exists(save_path+".index") and not ARGS.delete_prev_model_2:
                        saver.restore(session, save_path)
                        print("\tModel restored from: ", save_path)
                        if ARGS.skip_phase_2:
                            print("\tSkipping phase 2")
                            continue

                    start_time = time.time()
                    test_results = test(session)
                    append_results(test_results)
                    mse_new = test_results['accuracy']

                    while True:
                        t_epoch = t_epoch + 1
                        print("\tEpoch number: ", t_epoch)
                        mse_old = mse_new
                        train(session, optimizers[1], train_iters_adv_2)
                        test_results = test(session)
                        tot_iters_adv = tot_iters_adv + train_iters_adv_2
                        append_results(test_results)
                        write_results()
                        mse_new = test_results['accuracy']
                        # save model every 10 epochs
                        if t_epoch % 10 == 0:
                            saver.save(session, save_path)
                            print("\tModel saved in path: ", save_path)
                        # if a specific condition is satisfied, exit from the loop
                        condition = t_epoch >= ARGS.train_epochs_2
                        if ARGS.reach_convergence_2:
                            condition = abs(acc_old - acc_new) < ARGS.acc_epsilon_2
                        if condition:
                            break


                else:
                    # Phase 3: adversarial training of the network (minimax)
                    print("\tPhase 3: Adversarial training of the network (minimax).")
                    saver = tf.compat.v1.train.Saver()
                    save_path = save_dir + sim_slug + "/model.ckpt"

                    start_time = time.time()
                    test_results = test(session)
                    append_results(test_results)
                    mse_new = test_results['mse']
                    acc_new = test_results['accuracy']

                    while True:
                        t_epoch = t_epoch + 1
                        print("\tEpoch number: ", t_epoch)
                        mse_old = mse_new
                        acc_old = acc_new
                        train(session, optimizers[2], train_iters_legit_3)
                        test_results = test(session)
                        tot_iters_legit = tot_iters_legit + train_iters_legit_3
                        append_results(test_results)
                        write_results()
                        train(session, optimizers[1], train_iters_adv_3)
                        test_results = test(session)
                        tot_iters_adv = tot_iters_adv + train_iters_adv_3
                        append_results(test_results)
                        write_results()
                        acc_new = test_results['accuracy']
                        # save model every 10 epochs
                        if t_epoch % 10 == 0:
                            saver.save(session, save_path)
                            print("\tModel saved in path: ", save_path)
                        # if a specific condition is satisfied, exit from the loop
                        condition = t_epoch >= ARGS.train_epochs_3
                        if ARGS.reach_convergence_3:
                            condition = abs(mse_old - mse_new) < ARGS.mse_epsilon_3
                            condition = condition and abs(acc_old - acc_new) < ARGS.acc_epsilon_3
                        if condition:
                            break

    elif ARGS.mode in ("test", "Test"):
        DATASETS[ARGS.dataset].data_path = ARGS.data_dir_test
        DATASETS[ARGS.dataset].maybe_download_and_extract()

        # count the number of simulation folders
        num_simulations = 0
        folders = os.walk(save_dir)[1]
        for name in folders:
            if "sim" in name:
                num_simulations = num_simulations + 1

        for n in num_simulations:
            print("Beginning simulation number: ", n)
            sim_slug = "/sim{}".format(n)

            measures = {
                'snr_legit': [],
                'snr_adv': [],
                'mse': [],
                'psnr': [],
                'cross_entropy': [],
                'accuracy': [],
                'avg_power_y': [],
                'avg_power_z': []
            }

            def write_results():
                filename = test_dir+sim_slug+"/results.csv"

                df = pd.DataFrame(measures)
                df.to_csv(filename)
                print("\tResults saved in ", filename)

            def append_results(results):
                measures['snr_legit'].append(snr_legit)
                measures['snr_adv'].append(snr_adv)
                for key in results:
                    if key in measures:
                        if(t_epoch % 1 == 0):
                            print("\t -", key, ": {0}".format(results[key]))
                        measures[key].append(results[key])

            snr_range = [5*i for i in range(-5,6)] # [-25, -20, ..., 20, 25]

            snr_legit = ARGS.snr_legit_train
            for snr_adv in snr_range:

                model_vars, model_metrics, model_losses, model_collections = get_model(u, p,
                                                            params={ 'ARGS': ARGS,
                                                                     'snr_legit': snr_legit,
                                                                     'snr_adv': snr_adv})

                session = tf.Session(config = session_config)
                saver = tf.compat.v1.train.Saver()
                save_path = save_dir + sim_slug + "/model.ckpt"
                if os.path.exists(save_path+".index"):
                    saver.restore(session, save_path)
                    print("Model restored from: ", save_path)
                    test_results = test(session)
                    append_results(test_results)
                    write_results()


                else:
                    print("ERROR: Model not found.")



    else:
        print("Error: the only available options are 'train' or 'test'.")

if __name__ == '__main__':
    ARGS = args_parser.parse_args(DATASETS)
    run_main(ARGS)
    print('SUCCESS: Program ended correctly.')
