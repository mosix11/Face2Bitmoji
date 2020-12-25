import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import sys
sys.path.append('../')
import LACycleGAN.config as config

def conv_norm_relu_drop(inputs, filters, kernel_size, stride, initializer, norm_type=None, relu_type='lrelu', apply_dropout=False, use_bias=False):
    
    forw = inputs

    forw = keras.layers.Conv2D(filters, kernel_size, stride, padding='same', kernel_initializer=initializer, use_bias=use_bias)(forw)


    if norm_type=='batch':
        forw = keras.layers.BatchNormalization()(forw)
    elif norm_type=='instance':
        forw = tfa.layers.InstanceNormalization()(forw)
    
    if relu_type=='lrelu':
        forw = keras.layers.LeakyReLU(alpha=0.2)(forw)
    elif relu_type=='relu':
        forw = keras.layers.ReLU()(forw)

    if apply_dropout:
        forw = keras.layers.Dropout(rate=0.5)(forw)

    return forw

def trnsconv_norm_relu_drop(inputs, filters, kernel_size, stride, initializer, norm_type=None, relu_type='lrelu', apply_dropout=False):

    forw = keras.layers.Conv2DTranspose(filters, kernel_size, stride, padding='same', kernel_initializer=initializer, use_bias=False)(inputs)
    
    if norm_type=='batch':
        forw = keras.layers.BatchNormalization()(forw)
    elif norm_type=='instance':
        forw = tfa.layers.InstanceNormalization()(forw)
    
    if relu_type=='lrelu':
        forw = keras.layers.LeakyReLU(alpha=0.2)(forw)
    elif relu_type=='relu':
        forw = keras.layers.ReLU()(forw)

    if apply_dropout:
        forw = keras.layers.Dropout(rate=0.5)(forw)

    return forw


def conv_with_reflection_pad(inputs, filters, kernel_size, stride, initializer, use_bias=False):
    pad = int(np.floor(kernel_size / 2))
    forw = tf.pad(inputs, [[0,0], [pad,pad], [pad,pad], [0,0]], "REFLECT")
    if use_bias:
        forw = keras.layers.Conv2D(filters, kernel_size, stride, padding='valid', kernel_initializer=initializer, use_bias=True)(forw)
    else:
        forw = keras.layers.Conv2D(filters, kernel_size, stride, padding='valid', kernel_initializer=initializer, use_bias=False)(forw)
    return forw


def resnet_block(inputs, filters, initializer):

    forw = conv_with_reflection_pad(inputs, filters, 3, 1, initializer)
    forw = tfa.layers.InstanceNormalization()(forw)
    forw = keras.layers.ReLU()(forw)

    forw = conv_with_reflection_pad(forw, filters, 3, 1, initializer)
    forw = tfa.layers.InstanceNormalization()(forw)

    forw = keras.layers.Add()([forw, inputs])
    return forw


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = tf.cast(total_steps, tf.float32)
        self._step_decay = tf.cast(step_decay, tf.float32)
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate