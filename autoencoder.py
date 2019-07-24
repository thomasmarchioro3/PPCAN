import numpy as np
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool

import channel as ch

import sys
cifar_path = 'cifar_utils'
sys.path.insert(0, cifar_path)
import cifar10 as cf

image_width = cf.img_size
image_height = cf.img_size
image_channels = cf.num_channels

x_width = image_width/4
x_height = image_height/4

C_OUT = 16
CHANNEL_TYPE = ch.CHANNEL_TYPE
REAL_CHANNEL = ch.REAL_CHANNEL
CHANNEL_SNR = 12

xx = tf.placeholder(tf.float32, [None, x_width, x_height, C_OUT])
INTER_SHAPE = tf.shape(xx)

# topology of the NN
NN_PARAMS = {
    'filters_h1': 16,
    'kernel_size_h1': [5,5],
    'strides_h1': [2,2],
    
    'filters_h2': 32,
    'kernel_size_h2': [5,5],
    'strides_h2': [2,2],
    
    'filters_h3': 32,
    'kernel_size_h3': [5,5],
    'strides_h3': [1,1],
    
    'filters_h4': 32,
    'kernel_size_h4': [5,5],
    'strides_h4': [1,1],
    
    'filters_x': C_OUT,
    'kernel_size_x': [5,5],
    'strides_x': [1,1],
}


"""
    Helper functions
"""

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_dense_layer(tensor,      # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=False): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.add(tf.matmul(tensor, weights), biases)

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

"""
    Initializers
"""

kernel_initializer = lambda: tf.initializers.variance_scaling(2.0, 'fan_avg')
bias_initializer = lambda: tf.constant_initializer(0.0)

"""
    Parametric ReLU activation function:
    x         if x>=0
    alpha*x   if x<0
    
    Inputs:
    - x: input tensor (should be the output of a convolutional layer)
    - name: the name is needed for the scope of the variable alpha
"""

def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', 
                                x.get_shape()[-1],
                                x.dtype, 
                                tf.constant_initializer(0.25))
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

"""
    Returns a Tensorflow convolutional layer.
    
    Inputs:
    - inputs: The input tensor of the layer
    - filters: Integer, the dimensionality of the output space 
    - kernel_size: An integer or tuple/list of 2 integers, specifying 
    the height and width of the 2D convolution window; can be a single 
    integer to specify the same value for all spatial dimensions
    - strides: An integer or tuple/list of 2 integers, specifying the 
    strides of the convolution along the height and width; can be a 
    single integer to specify the same value for all spatial dimensions
    
    Important: tf.layers.conv2d is deprecated and will be replaced by 
    tf.keras.layers.Conv2D
"""

def conv_layer(inputs,
              filters,
              kernel_size,
              strides,
              activation=None
              ):
    return tf.layers.conv2d(
    #return tf.keras.layers.conv2D(
                            inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding="same",
                            data_format="channels_last",
                            activation=activation,
                            kernel_initializer = kernel_initializer(),
                            bias_initializer = bias_initializer())


def deconv_layer(inputs,
                filters,
                kernel_size,
                strides,
                activation=None
                ):
    return tf.layers.conv2d_transpose(
    #return tf.keras.layers.Conv2DTranspose(
                            inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding="same",
                            data_format="channels_last",
                            activation=activation,
                            kernel_initializer = kernel_initializer(),
                            bias_initializer = bias_initializer())


"""
    Returns a normalized version of the input tensor.
"""

def normalization_layer(x_in):
    if CHANNEL_TYPE == "awgn" and REAL_CHANNEL:
        dim_x = tf.shape(x_in)[1]
        # normalize latent vector so that the average power is 1
        x = (tf.sqrt(tf.cast(dim_x, dtype=tf.float32))
                * tf.nn.l2_normalize(x_in, axis=1))
    elif CHANNEL_TYPE == "fading" and not REAL_CHANNEL:
        # half of the channels are I component and half Q
        dim_x = tf.div(tf.shape(x_in)[1], 2)
        # normalization
        x = (tf.sqrt(tf.cast(dim_x, dtype=tf.float32))
                    * tf.nn.l2_normalize(x_in, axis=1))
    else:
        raise Exception("This option shouldn't be an option!")
    return x

"""
    ENCODER
"""

def encoder(inputs):
    h1 = conv_layer(inputs=inputs,
                   filters=NN_PARAMS['filters_h1'],
                   kernel_size=NN_PARAMS['kernel_size_h1'],
                   strides=NN_PARAMS['strides_h1']
                   )

    ph1 = prelu(h1, "ph1")

    h2 = conv_layer(inputs=ph1,
                   filters=NN_PARAMS['filters_h2'],
                   kernel_size=NN_PARAMS['kernel_size_h2'],
                   strides=NN_PARAMS['strides_h2'])

    ph2 = prelu(h2, "ph2")

    h3 = conv_layer(inputs=ph2,
                   filters=NN_PARAMS['filters_h3'],
                   kernel_size=NN_PARAMS['kernel_size_h3'],
                   strides=NN_PARAMS['strides_h3'])

    ph3 = prelu(h3, "ph3")

    h4 = conv_layer(inputs=ph3,
                   filters=NN_PARAMS['filters_h4'],
                   kernel_size=NN_PARAMS['kernel_size_h4'],
                   strides=NN_PARAMS['strides_h4'])

    ph4 = prelu(h4, "ph4")

    x_nf = conv_layer(inputs=ph4,
                   filters=NN_PARAMS['filters_x'],
                   kernel_size=NN_PARAMS['kernel_size_x'],
                   strides=NN_PARAMS['strides_x'])
    
    global INTER_SHAPE
    INTER_SHAPE = tf.shape(x_nf)
    
    x_flat = tf.layers.flatten(x_nf)
    x = normalization_layer(x_flat)
    
    return x


"""
    DECODER
    
    Inputs:
    - y: the input tensor of the decoder (output of the phyical channel)
"""

def decoder(y, dec_type=0, num_classes=10):
    
    y_reshaped = tf.reshape(y, INTER_SHAPE)
    
    y_expanded = deconv_layer(inputs=y_reshaped,
                      filters=NN_PARAMS['filters_h4'],
                      kernel_size=NN_PARAMS['kernel_size_x'],
                      strides=NN_PARAMS['strides_x'])

    py_expanded = prelu(y_expanded, "py_expanded")

    rev_h4 = deconv_layer(inputs=py_expanded,
                      filters=NN_PARAMS['filters_h3'],
                      kernel_size=NN_PARAMS['kernel_size_h4'],
                      strides=NN_PARAMS['strides_h4'])

    prev_h4 = prelu(rev_h4, "prev_h4")

    rev_h3 = deconv_layer(inputs=prev_h4,
                      filters=NN_PARAMS['filters_h2'],
                      kernel_size=NN_PARAMS['kernel_size_h3'],
                      strides=NN_PARAMS['strides_h3'])

    prev_h3 = prelu(rev_h3, "prev_h3")

    rev_h2 = deconv_layer(inputs=prev_h3,
                      filters=NN_PARAMS['filters_h1'],
                      kernel_size=NN_PARAMS['kernel_size_h2'],
                      strides=NN_PARAMS['strides_h2'])

    prev_h2 = prelu(rev_h2, "prev_h2")
    
    rev_h1 = deconv_layer(inputs=prev_h2,
                      filters=image_channels,
                      kernel_size=NN_PARAMS['kernel_size_h1'],
                      strides=NN_PARAMS['strides_h1'],
                      activation=tf.nn.sigmoid)
    
    m_hat = rev_h1
    
    return m_hat


"""
    SOFTMAX PREDICTOR
    
    Inputs:
    - m_hat: the output of the decoder
    - num_classes
"""

def soft_predictor(m_hat, num_classes=10):
    m_flat, num_features = flatten_layer(m_hat)
    
    size_1 = 128
    
    dense_1 = new_dense_layer(m_flat,
                             num_inputs=num_features,
                             num_outputs=size_1,
                             use_relu=True)  
    dense_2 = new_dense_layer(dense_1,
                             num_inputs=size_1,
                             num_outputs=num_classes,
                             use_relu=False)
    soft = tf.nn.softmax(dense_2)
    
    return soft
    

"""
    CHANNEL
"""

def channel(x, stddev):
    if CHANNEL_TYPE == "awgn" and REAL_CHANNEL:
        return ch.real_awgn(x, stddev)
    elif CHANNEL_TYPE == "fading" and not REAL_CHANNEL:
        return ch.fading(x, stddev)
    else:
        # by default do not introduce any noise
        return x
