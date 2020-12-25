import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import LACycleGAN.config as config

def load_image(path, dtype=tf.float32):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, dtype)
    return img

def resize(img, height, width, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    resized = tf.image.resize(img, [height, width], method)
    return resized

def random_crop(img, height, width):
    cropped_image = tf.image.random_crop(img, size=[tf.shape(img)[0], height, width, 3])
    return cropped_image

def normalize(img, current_min, current_max, desired_min, desired_max):
    img = (img - current_min) * ((desired_max - desired_min) / (current_max - current_min)) + desired_min
    return img

@tf.function
def normalize_tf(img, desired_min, desired_max):
    cur_min = tf.reduce_min(img)
    cur_max = tf.reduce_max(img)
    
    img = tf.multiply(tf.subtract(img, cur_min), tf.math.divide(desired_max-desired_min, cur_max-cur_min)) + desired_min
    return img
       

@tf.function()
def random_jitter_src_trgt(src_img, trgt_img, upscale_size):

    up_height, up_width = upscale_size
    current_shape = tf.shape(src_img)[1:]

    src_img = resize(src_img, up_height, up_width)
    trgt_img = resize(trgt_img, up_height, up_width)  


    src_img = random_crop(src_img, current_shape[0], current_shape[1])
    trgt_img = random_crop(trgt_img, current_shape[0], current_shape[1])

    if tf.random.uniform((), dtype=tf.float32) > 0.5:
        # random mirroring
        src_img = tf.image.flip_left_right(src_img)
        trgt_img = tf.image.flip_left_right(trgt_img)

    return src_img, trgt_img


@tf.function()
def random_jitter(img, upscale_size):

    up_height, up_width = upscale_size
    current_shape = tf.shape(img)[1:]

    img = resize(img, up_height, up_width)

    img = random_crop(img, current_shape[0], current_shape[1])

    if tf.random.uniform((), dtype=tf.float32) > 0.5:
        # random mirroring
        img = tf.image.flip_left_right(img)

    return img


def save_image(src_img, trgt_img, code):
    plt.figure(figsize=(8, 8))
    display_list = [src_img[0], trgt_img[0]]
    title = ['Input Image', 'Target Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    filename1 = config.TRAIN_SAMPLE_LOG_DIR + 'img_%06d.png' % (code+1)
    plt.savefig(filename1)
    plt.close()