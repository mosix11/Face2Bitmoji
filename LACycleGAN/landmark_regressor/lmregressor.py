import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sys

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

def trnsconv_norm_relu_drop(inputs, filters, kernel_size, stride, initializer, norm_type=None, relu_type='lrelu', apply_dropout=False, use_bias=False):

    forw = keras.layers.Conv2DTranspose(filters, kernel_size, stride, padding='same', kernel_initializer=initializer, use_bias=use_bias)(inputs)
    
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

def LandmarkRegressor():
    init = tf.random_normal_initializer(0., 0.02)
    # init = 'glorot_uniform'

    inputs = keras.layers.Input(shape=(128,128,3))

    conv1 = conv_norm_relu_drop(inputs, 64, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 64*64*64

    conv2 = conv_norm_relu_drop(conv1, 128, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 32*32*128

    conv3 = conv_norm_relu_drop(conv2, 256, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 16*16*256

    conv4 = conv_norm_relu_drop(conv3, 512, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 8*8*512

    conv5 = conv_norm_relu_drop(conv4, 1024, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 4*4*1024

    # Residual Block Output --> 4*4*1024
    res_conv = conv_norm_relu_drop(conv5, 1024, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True)
    res_deconv = trnsconv_norm_relu_drop(res_conv, 1024, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type=None, apply_dropout=False, use_bias=True)
    res_out = keras.layers.Add()([res_deconv, conv5])
    res_out = keras.layers.ReLU()(res_out)

    deconv5 = trnsconv_norm_relu_drop(res_out, 512, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=False) # Output --> 8*8*512
    deconv5 = keras.layers.Concatenate()([deconv5, conv4])

    deconv4 = trnsconv_norm_relu_drop(deconv5, 256, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=False) # Output --> 16*16*256
    deconv4 = keras.layers.Concatenate()([deconv4, conv3])    

    deconv3 = trnsconv_norm_relu_drop(deconv4, 128, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=False) # Output --> 32*32*128
    deconv3 = keras.layers.Concatenate()([deconv3, conv2]) 

    deconv2 = trnsconv_norm_relu_drop(deconv3, 64, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=False) # Output --> 64*64*64
    deconv2 = keras.layers.Concatenate()([deconv2, conv1]) 

    deconv1 = trnsconv_norm_relu_drop(deconv2, 32, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=False) # Output --> 128*128*32
    deconv1 = keras.layers.Concatenate()([deconv1, inputs]) 

    out = conv_norm_relu_drop(deconv1, 32, kernel_size=3, stride=1, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=False) # Output --> 128*128*32

    out = conv_norm_relu_drop(out, 3, kernel_size=3, stride=1, initializer=init, norm_type='batch', relu_type=None, apply_dropout=False, use_bias=True) # Output --> 128*128*3
    # out = keras.layers.Activation('sigmoid')(out)

    model = keras.Model(inputs, out)

    return model


# def LandmarkRegressor():
#     init = tf.random_normal_initializer(0., 0.02)
#     # init = 'glorot_uniform'

#     inputs = keras.layers.Input(shape=(128,128,3))

#     conv1 = conv_norm_relu_drop(inputs, 64, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 64*64*64

#     conv2 = conv_norm_relu_drop(conv1, 128, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 32*32*128

#     conv3 = conv_norm_relu_drop(conv2, 256, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='rely', apply_dropout=False, use_bias=True) # Output --> 16*16*256

#     conv4 = conv_norm_relu_drop(conv3, 512, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 8*8*512

#     conv5 = conv_norm_relu_drop(conv4, 1024, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 4*4*1024

#     conv6 = conv_norm_relu_drop(conv5, 2048, kernel_size=3, stride=2, initializer=init, norm_type='batch', relu_type='relu', apply_dropout=False, use_bias=True) # Output --> 2*2*2048

#     out = keras.layers.Flatten()(conv6)
#     out = keras.layers.Dense(10, kernel_initializer=init)(out) # Output --> 10*1

#     model = keras.Model(inputs, out)

#     return model

