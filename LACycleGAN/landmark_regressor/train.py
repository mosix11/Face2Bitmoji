import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import os 
import lmregressor
import time
import datetime


DATA_DIR = '../celeba2bitmoji/trainB/'
HEATMAP_DIR = '../celeba2bitmoji/trainB_lmheatmap/'
EPOCH = 400
BATCH_SIZE = 64
BUFFER_SIZE = 1024
CHECKPOINT_PATH = './checkpoints/cp.ckpt'

def load_image(path, dtype=tf.float32):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, dtype)
    return img

def normalize(img, current_min, current_max, desired_min, desired_max):
    img = (img - current_min) * ((desired_max - desired_min) / (current_max - current_min)) + desired_min
    return img

def process_img(path):
    img = load_image(path)
    img = normalize(img, 0, 255, -1, 1)
    return img


def save_image(x, y, preds, code):

    x = x[0].numpy()
    x = normalize(x, -1, 1, 0, 1)
    y = y[0].numpy()
    y = normalize(y, -1, 1, 0, 1)
    pred = preds[0].numpy()
    pred = normalize(pred, np.min(pred), np.max(pred), 0, 1)
    plt.figure(figsize=(15, 15))
    mixed_pred = np.clip(x + pred, 0, 1)
    mixed_trgt = np.clip(x + y, 0, 1)
    display_list = [x, y, mixed_trgt, pred, mixed_pred]
    title = ['Input Image', 'Target Image', 'Target Mixed Image', 'Predicted Image', 'Pred Mixed Image']

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    filename1 = './logs/' + 'img_%06d.png' % (code+1)
    plt.savefig(filename1)
    plt.close()



def load_train_dataset():

    x_dataset = tf.data.Dataset.list_files(DATA_DIR + '*.jpg', shuffle=False)
    y_dataset = tf.data.Dataset.list_files(HEATMAP_DIR + '*.jpg', shuffle=False)
    

    x_dataset = x_dataset.map(process_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y_dataset = y_dataset.map(process_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # for elem in x_dataset:
    #     plt.imshow(normalize(elem.numpy(), -1, 1, 0, 1))
    #     plt.show()


    # train_dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    # train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # return train_dataset

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset)).shuffle(BUFFER_SIZE)

    test_dataset = dataset.take(1).batch(BATCH_SIZE)

    valid_dataset = dataset.skip(1).take(1).batch(BATCH_SIZE, drop_remainder=False)

    train_dataset = dataset.skip(2).batch(BATCH_SIZE, drop_remainder=False)


    return test_dataset, valid_dataset, train_dataset





@tf.function
def mse(label, predicted):
    return tf.reduce_mean(tf.square(label - predicted))

@tf.function
def mae(label, predicted):
    return tf.reduce_mean(tf.abs(label - predicted))  

@tf.function
def train_step(model, x, y, optim):
    
    with tf.GradientTape() as tape:
        preds = model(x)
        # loss = mae(y, preds)
        loss = keras.losses.MSE(y, preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return {'loss':tf.reduce_mean(loss)}

def fit(model, train_dataset, valid_dataset, optim, epoch_counter, ckpt_manager, summary_writer):

    x_sample, y_sample = None, None
    for s in train_dataset.take(1):
        x_sample, y_sample = s

    with summary_writer.as_default():
        for e in range(epoch_counter.value(), EPOCH):
            # start = time.time()

            for b, (x_batch, y_batch) in enumerate(train_dataset):
                summary = train_step(model, x_batch, y_batch, optim)

                tf.summary.scalar('loss', summary['loss'], step=e)

                if (b+1) % 60 == 0:
                    valid_losses = []
                    for b, (x_batch, y_batch) in enumerate(valid_dataset):
                        preds = model(x_batch)
                        loss = keras.losses.MSE(y_batch, preds)
                        valid_losses.append(np.mean(loss.numpy()))

                    tf.summary.scalar('valid_loss', np.mean(valid_losses), step=e) 


            epoch_counter.assign_add(1)




            if e % 5 == 0:
                save_image(x_sample, y_sample, model(x_sample), e)
                for s in valid_dataset.take(1):
                    x_valid, y_valid = s
                save_image(x_valid, y_valid, model(x_valid), e+1)
                ckpt_manager.save()


if __name__ == '__main__':


    test_dataset, valid_dataset, train_dataset = load_train_dataset()

    model = lmregressor.LandmarkRegressor()

    epoch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)


    optim = keras.optimizers.Adam(learning_rate=0.0002)

    # ckpt = tf.train.Checkpoint(model=model,
    #                         optim=optim,
    #                         epoch_counter=epoch_counter)


    # ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=20)

    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print ('Latest checkpoint restored!!')

    # summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # fit(model, train_dataset, valid_dataset, optim, epoch_counter, ckpt_manager, summary_writer)


    # model.save('bitmoji_lmreg_heat')

    model = keras.models.load_model('bitmoji_lmreg_heat')

    valid_losses = []
    for b, (x_batch, y_batch) in enumerate(valid_dataset):
        preds = model(x_batch)
        loss = keras.losses.MSE(y_batch, preds)
        valid_losses.append(np.mean(loss.numpy()))


    print(valid_losses)
    print(np.mean(valid_losses))

    test_losses = []
    for b, (x_batch, y_batch) in enumerate(test_dataset):
        preds = model(x_batch)
        loss = keras.losses.MSE(y_batch, preds)
        test_losses.append(np.mean(loss.numpy()))


    print(test_losses)
    print(np.mean(test_losses))