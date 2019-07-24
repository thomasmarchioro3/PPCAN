import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

CHANNEL_TYPE = "awgn"
REAL_CHANNEL = True

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

def fading(x, stddev):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    h = tf.complex(tf.random_normal([tf.shape(x)[0], 1], 0, 1/np.sqrt(2)),
                   tf.random_normal([tf.shape(x)[0], 1], 0, 1/np.sqrt(2)))

    # additive white gaussian noise
    awgn = tf.complex(tf.random_normal(tf.shape(x), 0, 1/np.sqrt(2)),
                      tf.random_normal(tf.shape(x), 0, 1/np.sqrt(2)))

    return h*x + stddev*awgn

def fake_fading(x, stddev):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise without converting to complex.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    # h = tfp.math.random_rayleigh(shape=[tf.shape(x)[0], 1],
    #                              scale=1/np.sqrt(2),
    #                              dtype=tf.float32)
    # h = tf.minimum(h, 1e13)
    n1 = tf.random_normal([tf.shape(x)[0], 1], 0, 1/np.sqrt(2),
                          dtype=tf.float32)
    n2 = tf.random_normal([tf.shape(x)[0], 1], 0, 1/np.sqrt(2),
                          dtype=tf.float32)

    h = tf.sqrt(tf.square(n1)+tf.square(n2))

    # additive white gaussian noise
    awgn = tf.random_normal(tf.shape(x), 0,
                            stddev/np.sqrt(2),
                            dtype=tf.float32)

    return h*x + awgn