import tensorflow as tf
import numpy as np
from autoencoder import adversarial_networks
import cifar10

DATASETS = {"cifar": cifar10}

def equalization_matrix(cluster_vector):
    num_input = len(cluster_vector)
    E = np.zeros((num_input,num_input))
    for j in range(num_input):
        for i in range(num_input):
            if (cluster_vector[i] == cluster_vector[j]):
                E[i,j]=1
    E = E/np.sum(E, axis=1) # normalize
    return E

def softmax_equalizer(p, num_classes, num_clusters=1):
    if(num_clusters <= 0):
        num_clusters = 1

    cluster_vector = []

    cluster_size = int(num_classes/num_clusters)

    label = 0

    for i in range(num_classes):
        if label < num_classes -1 and i > (label+1)*cluster_size:
            label = label + 1
        cluster_vector.append(label)
    cluster_vector = np.random.permutation(cluster_vector)
    #print("CLUSTER VECTOR: ", cluster_vector)
    E = equalization_matrix(cluster_vector)
    #print("EQUALIZATION MATRIX: ", E)
    p_eq = tf.matmul(p, tf.cast(tf.convert_to_tensor(E), tf.float32))
    return p_eq



def get_model(u, p, params):
    ARGS = params['ARGS']

    snr_legit = ARGS.snr_legit_train
    snr_adv = ARGS.snr_adv_train
    if ARGS.mode not in ("train", "Train"):
        snr_legit = params['snr_legit_test']
        snr_adv = params['snr_adv_test']

    num_classes = DATASETS[ARGS.dataset]._NUM_CLASSES
    #p = tf.placeholder(tf.float32, shape=[None, num_classes], name='p')
    s = tf.argmax(p, axis=1)

    tensors = adversarial_networks(u, ARGS, snr_legit, snr_adv)

    u_hat = tensors['u_hat']
    q_tilde = tensors['q_tilde']
    s_tilde = tensors['s_tilde']
    avg_power_y = tensors['avg_power_y']
    avg_power_z = tensors['avg_power_z']

    # loss of the legitimate channel
    mse = tf.compat.v1.losses.mean_squared_error(u, u_hat)
    psnr_batch = tf.image.psnr(u, u_hat, max_val=1.0)
    avg_psnr_batch = tf.reduce_mean(psnr_batch, name="psnr_avg_batch")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=q_tilde, labels=p)
    avg_cross_entropy = tf.reduce_mean(cross_entropy, name="cross_avg_batch")

    correct_prediction = tf.equal(s_tilde, s)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    p_eq = softmax_equalizer(p, num_classes=num_classes, num_clusters=ARGS.num_clusters)
    cross_entropy_eq = tf.nn.softmax_cross_entropy_with_logits_v2(logits=q_tilde, labels=p_eq)
    avg_cross_entropy_eq = tf.reduce_mean(cross_entropy_eq, name="cross_avg_batch_eq")

    #print("Tensor p: ", p)
    #print("Tensor p_eq: ", p_eq)
    #print("Tensor q_tilde: ", q_tilde)

    loss_legit_prelim = mse
    loss_adv = avg_cross_entropy

    alpha = ARGS.alpha

    # default loss type is 2
    loss_legit = mse + alpha*avg_cross_entropy_eq
    if(ARGS.loss_type == 1):
        loss_legit = mse - alpha*avg_cross_entropy

    legitimate_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'legitimate')
    adversary_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         'adversary')


    model_vars = {
        'u': u,
        's': s,
        'p': p,
        'p_eq': p_eq,
        'u_hat': u_hat,
        's_tilde': s_tilde,
        'q_tilde': q_tilde
    }
    model_metrics = {
        'mse': mse,
        'psnr': avg_psnr_batch,
        'cross_entropy': avg_cross_entropy,
        'cross_entropy_eq': avg_cross_entropy_eq,
        'accuracy': accuracy,
        'avg_power_y': avg_power_y,
        'avg_power_z': avg_power_z
    }
    model_losses = {
        'loss_legit_prelim': loss_legit_prelim,
        'loss_adv': loss_adv,
        'loss_legit': loss_legit
    }

    model_collections = {
        'legitimate_vars': legitimate_vars,
        'adversary_vars': adversary_vars
    }

    return model_vars, model_metrics, model_losses, model_collections
