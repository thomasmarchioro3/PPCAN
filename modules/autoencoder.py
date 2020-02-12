import cifar10
import tensorflow as tf
import numpy as np

DATASETS = {"cifar": cifar10}

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def dense_layer(tensor,      # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=False): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

    layer = tf.add(tf.matmul(tensor, weights), biases)

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.random_normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn

    return y

def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', x.get_shape()[-1],
                                x.dtype, tf.constant_initializer(0.25))
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def adversarial_networks(u, ARGS, snr_legit, snr_adv):

    c_out = ARGS.conv_depth
    num_channels = DATASETS[ARGS.dataset]._NUM_CHANNELS
    num_classes = DATASETS[ARGS.dataset]._NUM_CLASSES

    def encoder(u):
        data_format = 'channels_last'
        conv_k_init = lambda: tf.initializers.variance_scaling(2.0, 'fan_avg')
        conv_b_init = lambda: tf.constant_initializer(0.0)
        h1 = tf.layers.conv2d(u, 16, 5, 2,
                              padding="same", data_format=data_format,
                              kernel_initializer=conv_k_init(),
                              bias_initializer=conv_b_init())
        ph1 = prelu(h1, "ph1")
        h2 = tf.layers.conv2d(ph1, 32, 5, 2,
                              padding="same", data_format=data_format,
                              kernel_initializer=conv_k_init(),
                              bias_initializer=conv_b_init())
        ph2 = prelu(h2, "ph2")
        h3 = tf.layers.conv2d(ph2, 32, 5, 1,
                              padding="same", data_format=data_format,
                              kernel_initializer=conv_k_init(),
                              bias_initializer=conv_b_init())
        ph3 = prelu(h3, "ph3")
        h4 = tf.layers.conv2d(ph3, 32, 5, 1,
                              padding="same", data_format=data_format,
                              kernel_initializer=conv_k_init(),
                              bias_initializer=conv_b_init())
        ph4 = prelu(h4, "ph4")

        # The number of channel outputs (c_out) define the compression rate
        # At this point, h4 output has a tensor H/4,W/4,32, and input is H,W,3
        # input: HxWx3
        # Therefore, the compression rate k/n can be computed as:
        # k/n = (H/4xW/4xc_out)/(HxWx3) = c_out/(16*3)
        x = tf.layers.conv2d(ph4, c_out, 5, 1,
                             padding="same", data_format=data_format)
        return x

    def channel(x, snr):
        inter_shape = tf.shape(x)
        # reshape array to [-1, dim_z]args
        y = tf.layers.flatten(x)

        # convert from snr to std
        noise_stddev = np.sqrt(10**(-snr/10))

        # Add channel noise
        dim_y = tf.shape(y)[1]
        # normalize latent vector so that the average power is 1
        y_in = (tf.sqrt(tf.cast(dim_y, dtype=tf.float32))
                * tf.nn.l2_normalize(y, axis=1))
        y_out = real_awgn(y_in, noise_stddev)

        # convert signal back to intermediate shape
        y = tf.reshape(y_in, inter_shape)
        # compute average power
        avg_power = tf.reduce_mean(tf.real(y_in*tf.conj(y_in)))
        return y, avg_power

    def decoder(y):
        data_format = 'channels_last'
        deconv_k_init = lambda: tf.initializers.variance_scaling(2.0, 'fan_avg')
        deconv_b_init = lambda: tf.constant_initializer(0.0)
        y_expanded = tf.layers.conv2d(y, 32, 5, 1,
                                      padding="same",
                                      data_format=data_format)
        py_expanded = prelu(y_expanded, "py_expanded")

        rev_h4 = tf.layers.conv2d_transpose(py_expanded,
                                            32, 5, 1,
                                            padding="same",
                                            data_format=data_format,
                                            kernel_initializer=deconv_k_init(),
                                            bias_initializer=deconv_b_init())
        prev_h4 = prelu(rev_h4, "prev_h4")
        rev_h3 = tf.layers.conv2d_transpose(prev_h4,
                                            32, 5, 1,
                                            padding="same",
                                            data_format=data_format,
                                            kernel_initializer=deconv_k_init(),
                                            bias_initializer=deconv_b_init())
        prev_h3 = prelu(rev_h3, "prev_h3")
        rev_h2 = tf.layers.conv2d_transpose(prev_h3,
                                            16, 5, 2,
                                            padding="same",
                                            data_format=data_format,
                                            kernel_initializer=deconv_k_init(),
                                            bias_initializer=deconv_b_init())
        prev_h2 = prelu(rev_h2, "prev_h2")
        rev_h1 = tf.layers.conv2d_transpose(prev_h2,
                                            num_channels, 5, 2,
                                            padding="same",
                                            data_format=data_format,
                                            kernel_initializer=deconv_k_init(),
                                            bias_initializer=deconv_b_init(),
                                            activation=tf.nn.sigmoid)
        u_hat = rev_h1
        return u_hat

    def soft_predictor(u_hat, num_classes):
        u_flat, num_features = flatten_layer(u_hat)
        size_1 = 128
        dense_1 = dense_layer(u_flat,
                                 num_inputs=num_features,
                                 num_outputs=size_1,
                                 use_relu=True)
        dense_2 = dense_layer(dense_1,
                                 num_inputs=size_1,
                                 num_outputs=num_classes,
                                 use_relu=False)
        q_tilde = tf.nn.softmax(dense_2)

        return q_tilde

    def legitimate_network(u):
        with tf.variable_scope('legitimate', reuse=tf.AUTO_REUSE):
            x = encoder(u)
            y, avg_power_y = channel(x, snr_legit)
            u_hat = decoder(y)
        return x, u_hat, avg_power_y

    def adversary_network(x):
        with tf.variable_scope('adversary', reuse=tf.AUTO_REUSE):
            z, avg_power_z = channel(x, snr_adv)
            u_tilde = decoder(z)
            q_tilde = soft_predictor(u_tilde, num_classes=num_classes)
            s_tilde = tf.argmax(q_tilde, axis=1)
        return q_tilde, s_tilde, avg_power_z

    x, u_hat, avg_power_y = legitimate_network(u)
    q_tilde, s_tilde, avg_power_z = adversary_network(x)

    tensors = {
        "u_hat": u_hat,
        "q_tilde": q_tilde,
        "s_tilde": s_tilde,
        "avg_power_y": avg_power_y,
        "avg_power_z": avg_power_z
    }

    return tensors
