import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import model.modules as modules
import sys
sys.path.append('../')
import LACycleGAN.config as config



def UnconditionalGlobalDiscriminator():

    init = tf.random_normal_initializer(stddev=0.02)

    img_input = keras.layers.Input(shape=config.IMG_SHAPE)

    forw = modules.conv_norm_relu_drop(img_input, 64, kernel_size=4, stride=2, initializer=init, relu_type='lrelu', apply_dropout=False) # 64*64*64

    forw = modules.conv_norm_relu_drop(forw, 128, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 32*32*128

    forw = modules.conv_norm_relu_drop(forw, 256, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 16*16*256

    # forw = conv_norm_relu_drop(forw, 512, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 8*8*512

    forw = modules.conv_norm_relu_drop(forw, 512, kernel_size=4, stride=1, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 16*16*512

    patch_out = modules.conv_norm_relu_drop(forw, 1, kernel_size=4, stride=1, initializer=init, norm_type=None, relu_type=None, apply_dropout=False, use_bias=True) # 16*16*1

    model = keras.Model(img_input, patch_out)

    return model

# def UnconditionalGlobalDiscriminator():

#     init = tf.random_normal_initializer(stddev=0.02)

#     img_input = keras.layers.Input(shape=config.IMG_SHAPE)

#     gndr_input = keras.layers.Input(shape=(1,))

#     embed = keras.layers.Embedding(2, 50)(gndr_input)
#     embed = keras.layers.Dense(config.IMG_SHAPE[0] * config.IMG_SHAPE[1], kernel_initializer=init)(embed)
#     embed = keras.layers.Reshape((config.IMG_SHAPE[0], config.IMG_SHAPE[1], 1))(embed)

#     merged = keras.layers.Concatenate()([img_input, embed])    

#     forw = modules.conv_norm_relu_drop(merged, 64, kernel_size=4, stride=2, initializer=init, relu_type='lrelu', apply_dropout=False) # 64*64*64

#     forw = modules.conv_norm_relu_drop(forw, 128, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 32*32*128

#     forw = modules.conv_norm_relu_drop(forw, 256, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 16*16*256

#     # forw = conv_norm_relu_drop(forw, 512, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 8*8*512

#     forw = modules.conv_norm_relu_drop(forw, 512, kernel_size=4, stride=1, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 16*16*512

#     patch_out = modules.conv_norm_relu_drop(forw, 1, kernel_size=4, stride=1, initializer=init, norm_type=None, relu_type=None, apply_dropout=False, use_bias=True) # 16*16*1

#     model = keras.Model([img_input, gndr_input], patch_out)

#     return model

def ConditionalGlobalDiscriminator():
    
    init = tf.random_normal_initializer(stddev=0.02)

    img_input = keras.layers.Input(shape=config.IMG_SHAPE)
    lm_input = keras.layers.Input(shape=config.IMG_SHAPE)

    merged_input = keras.layers.Concatenate()([img_input, lm_input])

    forw = modules.conv_norm_relu_drop(merged_input, 64, kernel_size=4, stride=2, initializer=init, norm_type=None, relu_type='lrelu', apply_dropout=False) # 64*64*64

    forw = modules.conv_norm_relu_drop(forw, 128, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 32*32*128

    forw = modules.conv_norm_relu_drop(forw, 256, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 16*16*256

    forw = modules.conv_norm_relu_drop(forw, 512, kernel_size=4, stride=2, initializer=init, norm_type='instance', relu_type='lrelu', apply_dropout=False) # 8*8*512

    forw = keras.layers.Flatten()(forw)
    
    out = keras.layers.Dense(1, kernel_initializer=init)(forw)

    model = keras.Model([img_input, lm_input], out)

    return model


# def Generator():
    
#     init = tf.random_normal_initializer(stddev=0.02)
    
#     img_input = keras.layers.Input(shape=config.IMG_SHAPE)
#     gndr_input = keras.layers.Input(shape=(1,))

#     embed = keras.layers.Embedding(2, 50)(gndr_input)
#     embed = keras.layers.Dense(config.IMG_SHAPE[0] * config.IMG_SHAPE[1], kernel_initializer=init)(embed)
#     embed = keras.layers.Reshape((config.IMG_SHAPE[0], config.IMG_SHAPE[1], 1))(embed)

#     merged = keras.layers.Concatenate()([img_input, embed])

#     forw = modules.conv_with_reflection_pad(merged, 64, kernel_size=7, stride=1, initializer=init)
#     forw = tfa.layers.InstanceNormalization()(forw)
#     forw = keras.layers.ReLU()(forw)


#     forw = modules.conv_norm_relu_drop(forw, 128, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)

#     forw = modules.conv_norm_relu_drop(forw, 256, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)

#     for i in range(config.RESNET_BLOCKS):
#         forw = modules.resnet_block(forw, 256, init)

    
#     forw = modules.trnsconv_norm_relu_drop(forw, 128, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)

#     forw = modules.trnsconv_norm_relu_drop(forw, 64, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)


#     forw = modules.conv_with_reflection_pad(forw, 3, kernel_size=7, stride=1, initializer=init, use_bias=True)

#     if config.SKIP:
#         forw = keras.layers.Add()([forw, img_input])

#     out = keras.layers.Activation('tanh')(forw)

#     # model = keras.Model(img_input, out)
#     model = keras.Model([img_input, gndr_input], out)

#     return model


def Generator():
    
    init = tf.random_normal_initializer(stddev=0.02)
    
    img_input = keras.layers.Input(shape=config.IMG_SHAPE)
    lm_input = keras.layers.Input(shape=config.IMG_SHAPE)

    merged = keras.layers.Concatenate()([img_input, lm_input])

    forw = modules.conv_with_reflection_pad(merged, 64, kernel_size=7, stride=1, initializer=init)
    forw = tfa.layers.InstanceNormalization()(forw)
    forw = keras.layers.ReLU()(forw)


    forw = modules.conv_norm_relu_drop(forw, 128, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)

    forw = modules.conv_norm_relu_drop(forw, 256, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)

    for i in range(config.RESNET_BLOCKS):
        forw = modules.resnet_block(forw, 256, init)
    
    forw = modules.trnsconv_norm_relu_drop(forw, 128, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)

    forw = modules.trnsconv_norm_relu_drop(forw, 64, kernel_size=3, stride=2, initializer=init, norm_type='instance', relu_type='relu', apply_dropout=False)


    forw = modules.conv_with_reflection_pad(forw, 3, kernel_size=7, stride=1, initializer=init, use_bias=True)

    if config.SKIP:
        forw = keras.layers.Add()([forw, img_input])

    out = keras.layers.Activation('tanh')(forw)

    model = keras.Model([img_input, lm_input], out)

    return model


# mse = tf.losses.MeanSquaredError()
# mae = tf.losses.MeanAbsoluteError()

# def discriminator_adv_loss(real_labels, generated_labels):  
#     real_loss = mse(tf.ones_like(real_labels), real_labels)
#     generated_loss = mse(tf.zeros_like(generated_labels), generated_labels)
#     return real_loss * 0.5 + generated_loss * 0.5

# def generator_adv_loss(generated_labels):
#     generator_loss = mse(tf.ones_like(generated_labels), generated_labels)
#     return generator_loss

# def identity_loss(src_img, generated_img):
#     identity_loss = mae(src_img, generated_img)
#     return config.IDENTITY_LOSS_LAMBDA * identity_loss

# def cycle_consistency_loss(src_img, cycled_img):
#     cycle_consistency_loss = mae(src_img, cycled_img)
#     return config.LAMBDA * cycle_consistency_loss
bce = keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def discriminator_adv_loss(real_labels, generated_labels):
    real_loss = tf.reduce_mean(tf.square(real_labels - tf.ones_like(real_labels)))
    generated_loss = tf.reduce_mean(tf.square(generated_labels - tf.zeros_like(generated_labels)))
    return (real_loss + generated_loss) * config.LAMBDA_UGD

# @tf.function
# def discriminator_adv_loss_CGD(real_labels, generated_labels1):
#     real_loss = bce(tf.ones_like(real_labels), real_labels)
#     generated_loss = bce(tf.zeros_like(generated_labels1), generated_labels1)
#     return (real_loss + generated_loss) * config.LAMBDA_CGD

@tf.function
def discriminator_adv_loss_CGD(real_labels, generated_labels1, generated_labels2):
    real_loss = tf.reduce_mean(tf.square(real_labels - tf.ones_like(real_labels)))
    generated_loss1 = tf.reduce_mean(tf.square(generated_labels1 - tf.zeros_like(generated_labels1)))
    generated_loss2 = tf.reduce_mean(tf.square(generated_labels2 - tf.zeros_like(generated_labels2)))
    return (real_loss + (generated_loss1 + generated_loss2) * 0.5) * config.LAMBDA_CGD

@tf.function
def generator_adv_loss(generated_labels):
    generator_loss = tf.reduce_mean(tf.square(generated_labels - tf.ones_like(generated_labels)))
    return generator_loss

@tf.function
def identity_loss(src_img, generated_img):
    identity_loss = tf.reduce_mean(tf.abs(src_img - generated_img))
    return config.LAMBDA_ID * identity_loss

@tf.function
def cycle_consistency_loss(src_img, cycled_img):
    cycle_consistency_loss = tf.reduce_mean(tf.abs(src_img - cycled_img))
    return config.LAMBDA_CYC * cycle_consistency_loss

@tf.function
def landmark_consistency_loss(real_landmarks, predicted_landmarks):
    lm_loss = tf.reduce_mean(tf.square(real_landmarks - predicted_landmarks))
    return lm_loss * config.LAMBDA_LM


# bce = keras.losses.BinaryCrossentropy(from_logits=True)
# @tf.function
# def auxiliary_loss(real_labels, predicted_labels):
#     return bce(real_labels, predicted_labels) * config.LAMBDA_UGD