import numpy as np
import tensorflow as tf
import data.image_util as im_util
import sys
sys.path.append('../')
import LACycleGAN.config as config


def load_image_train(path):
    img = im_util.load_image(path)
    # img = im_util.resize(img, config.IMG_HEIGHT, config.IMG_WIDTH)
    img = im_util.normalize(img, 0, 255, -1, 1)
    return img

def load_image_test(path):
    img = im_util.load_image(path)
    # img = im_util.resize(img, config.IMG_HEIGHT, config.IMG_WIDTH)
    img = im_util.normalize(img, 0, 255, -1, 1)
    return img


def load_train_dataset():
    a_imgs = tf.data.Dataset.list_files(config.TRAIN_A_DIR + '*.jpg', shuffle=False)
    a_lmhm = tf.data.Dataset.list_files(config.TRAIN_A_LM_DIR + '*.jpg', shuffle=False)

    b_imgs = tf.data.Dataset.list_files(config.TRAIN_B_DIR + '*.jpg', shuffle=False)
    b_lmhm = tf.data.Dataset.list_files(config.TRAIN_B_LM_DIR + '*.jpg', shuffle=False)
    

    a_gndrs = np.load(config.TRAIN_A_GENDER_LABELS)
    a_gndrs[a_gndrs == -1] = 0
    a_gndrs = tf.data.Dataset.from_tensor_slices(a_gndrs)

    b_gndrs = np.load(config.TRAIN_B_GENDER_LABELS)
    b_gndrs[b_gndrs == -1] = 0
    b_gndrs = tf.data.Dataset.from_tensor_slices(b_gndrs)

    a_imgs = a_imgs.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    a_lmhm = a_lmhm.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    b_imgs = b_imgs.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    b_lmhm = b_lmhm.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    a_dataset = tf.data.Dataset.zip((a_imgs, a_lmhm, a_gndrs))
    b_dataset = tf.data.Dataset.zip((b_imgs, b_lmhm, b_gndrs))


    a_dataset = a_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE, drop_remainder=True)
    b_dataset = b_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE, drop_remainder=True)

    return a_dataset, b_dataset


# def load_test_dataset():
#     a_dataset = tf.data.Dataset.list_files(config.TEST_A_DIR + '*.jpg')
#     b_dataset = tf.data.Dataset.list_files(config.TEST_B_DIR + '*.jpg')
    
#     a_dataset = a_dataset.map(load_image_test)
#     b_dataset = b_dataset.map(load_image_test)

#     a_dataset = a_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
#     b_dataset = b_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)

#     return a_dataset, b_dataset


def zip_datasets(a_set, b_set):
    a_card, b_card = a_set.cardinality(), b_set.cardinality()
    zipped = None

    if a_card > b_card:
        a_sh = a_set.shuffle(config.BUFFER_SIZE)
        b_sh = b_set.shuffle(config.BUFFER_SIZE)
        b_sh = b_sh.repeat()
        zipped = tf.data.Dataset.zip((a_sh, b_sh))

    elif a_card < b_card:
        a_sh = a_set.shuffle(config.BUFFER_SIZE)
        b_sh = b_set.shuffle(config.BUFFER_SIZE)        
        a_sh = a_sh.repeat()
        zipped = tf.data.Dataset.zip((a_sh, b_sh))

    else:
        a_sh = a_set.shuffle(config.BUFFER_SIZE)
        b_sh = b_set.shuffle(config.BUFFER_SIZE)
        zipped = tf.data.Dataset.zip((a_sh, b_sh))

    return zipped
    

class ItemPool:

    def __init__(self):
        self.pool_size = config.ITEMPOOL_SIZE
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch dataset tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)