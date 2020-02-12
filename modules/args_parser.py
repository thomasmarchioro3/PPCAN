import argparse

def positive(type_func):
    def check_positive(string):
        value = type_func(string)
        if value <= 0:
            raise argparse.ArgumentTypeError("value should be positive")
        return value
    return check_positive

def str2bool(v):
  return v.lower() in ("yes", "Yes", "true", "True", "t")

def parse_args(dataset):
    """parsing and configuration"""

    desc = "Adversarial Networks for Privacy Preserving Communications"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='train',
                        help="Can be either 'train' ot 'test'")

    parser.add_argument('--num_simulations', type=positive(int), default=5,
                        help="Number of complete trainings of the NN")

    # Preliminary training of main network

    parser.add_argument('--reach_convergence_1', type=str2bool, default=False,
                        help="If set to True, the main network is trained "
                        "until convergence in the preliminary training phase")

    parser.add_argument('--mse_epsilon_1', type=float, default=1e-4,
                        help="If the MSE variation is below this value, "
                             "convergence is reached")

    parser.add_argument('--train_epochs_1', type=positive(int), default=1,
                        help="Number of epochs of the main network in "
                             "preliminary training")

    parser.add_argument('--train_iters_legit_1', type=positive(int),
                        default=30000,
                        help="Number of preliminary iterations of the "
                        " main network")

    parser.add_argument('--skip_phase_1', type=str2bool, default=False,
                        help=('Skip training in phase 1 if model alreadt exists.'))

    parser.add_argument('--delete_prev_model_1', type=str2bool, default=False,
                        help=('Delete previuos model trained in phase 1.'))

    # Preliminary training of adversary's network

    parser.add_argument('--reach_convergence_2', type=str2bool, default=False,
                        help="If set to True, the adversary is trained until "
                        "convergence in the preliminary training phase")

    parser.add_argument('--acc_epsilon_2', type=float, default=1e-3,
                        help="If the accuracy variation is below this value, "
                             "convergence is reached")

    parser.add_argument('--train_epochs_2', type=positive(int), default=1,
                        help="Number of epochs of the adversary in "
                             "preliminary training")

    parser.add_argument('--train_iters_adv_2', type=positive(int),
                        default=30000,
                        help="Number of preliminary iterations of the "
                        " adversary network")

    parser.add_argument('--skip_phase_2', type=str2bool, default=False,
                        help=('Skip training in phase 2 if model alreadt exists.'))

    parser.add_argument('--delete_prev_model_2', type=str2bool, default=False,
                        help=('Delete previuos model trained in phase 2'))

    # Adversarial training of the network

    parser.add_argument('--reach_convergence_3', type=str2bool, default=True,
                        help="If set to True, the network is trained "
                        "until convergence in the adversarial phase")

    parser.add_argument('--mse_epsilon_3', type=float, default=1e-4,
                        help="If the MSE variation is below this value, "
                             "convergence is reached")

    parser.add_argument('--acc_epsilon_3', type=float, default=1e-3,
                        help="If the accuracy variation is below this value, "
                             "convergence is reached")

    parser.add_argument('--train_epochs_3', type=positive(int), default=40,
                        help="Number of epochs in the adversarial training")

    parser.add_argument('--train_iters_legit_3', type=positive(int), default=500,
                        help="Number of preliminary iterations of the "
                        " two networks")

    parser.add_argument('--train_iters_adv_3', type=positive(int), default=4000,
                        help="Number of iterations of the adversary "
                             "network per epoch")

    # other network paramenters

    parser.add_argument('--conv_depth', type=positive(float), default=16,
                        help="Number of channels of last conv layer, used to "
                              "define the compression rate: k/n=c_out/(16*3)")

    parser.add_argument('--snr_legit_train', type=positive(float), default=10,
                        help="SNR of the legitimate channel (dB) "
                             "on training")

    parser.add_argument('--snr_adv_train', type=positive(float), default=5,
                        help=("SNR of the adversary's channel (dB) "
                             "on training"))

    parser.add_argument('--snr_legit_test', type=positive(float), default=10,
                        help=("SNR of the legitimate channel (dB) "
                             "on test"))

    parser.add_argument('--snr_adv_test', type=positive(float), default=5,
                        help=("SNR of the adversary's channel (dB) "
                             "on test"))

    # other training paramenters

    parser.add_argument('--loss_type', type=positive(int), default=2,
                        help="Set 1 for cross-entropy maximization, 2 "
                             "for softmax equalization")

    parser.add_argument('--alpha', type=positive(float), default=1,
                        help=("Privacy-quality tradeoff parameter: "
                             "1) Loss = MSE + alpha*E[log(q_tilde)]"
                             "2) Loss = MSE + alpha*H(p_eq, q_tilde)"))

    parser.add_argument('--learn_rate', type=positive(float), default=1e-4,
                        help='Learning rate for Adam optimizer')

    parser.add_argument('--batch_size', type=positive(int), default=32,
                        help='Batch size')

    # softmax equalization

    parser.add_argument('--num_clusters', type=positive(int), default=1,
                        help="Number of clusters for softmax equalization, "
                             "required only if loss_type is set to 1")

    # dataset and directories

    parser.add_argument('--dataset', type=str, default='cifar',
                        choices=dataset.keys(),
                        help=('Choose image dataset.'
                              ' Options: {}'.format(dataset.keys())))

    parser.add_argument('--train_dir', type=str, default='results',
                        help=('The location of the model checkpoint files.'))

    parser.add_argument('--test_dir', type=str, default='results',
                        help=('The location of test files '
                              '(tensorboard, etc).'))

    parser.add_argument('--data_dir_train', type=str, default='data',
                        help='Directory where to store the training data set')

    parser.add_argument('--data_dir_test', type=str, default='data',
                        help='Directory where to store the test data set')


    return parser.parse_args()
