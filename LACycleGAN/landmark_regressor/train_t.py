import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import os 
import lmregressor
import time
import datetime
import cv2


DATA_DIR = '../celeba2bitmoji/trainB/'
LANDAMRKS_PATH = '../celeba2bitmoji/bitmoji_landmarks.npy'
EPOCH = 200
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
    img = normalize(img, 0, 255, 0, 1)
    return img

def save_image(src_img, y, preds, code):

    y_map = np.zeros_like(src_img[0])
    color = (255,255,255)
    y_land = y[0].numpy()
    cv2.circle(y_map, (y_land[0], y_land[1]), 2, color, 2)
    cv2.circle(y_map, (y_land[2], y_land[3]), 2, color, 2)
    cv2.circle(y_map, (y_land[4], y_land[5]), 2, color, 2)
    cv2.circle(y_map, (y_land[6], y_land[7]), 2, color, 2)
    cv2.circle(y_map, (y_land[8], y_land[9]), 2, color, 2)


    pred_map = np.zeros_like(src_img[0])
    color = (255,255,255)
    pred_land = preds[0].numpy()
    cv2.circle(pred_map, (pred_land[0], pred_land[1]), 2, color, 2)
    cv2.circle(pred_map, (pred_land[2], pred_land[3]), 2, color, 2)
    cv2.circle(pred_map, (pred_land[4], pred_land[5]), 2, color, 2)
    cv2.circle(pred_map, (pred_land[6], pred_land[7]), 2, color, 2)
    cv2.circle(pred_map, (pred_land[8], pred_land[9]), 2, color, 2)

    plt.figure(figsize=(15, 15))
    mixed_pred = np.clip(src_img[0] + pred_map, 0, 1)
    mixed_trgt = np.clip(src_img[0] + y_map, 0, 1)
    display_list = [src_img[0], y_map, mixed_trgt, pred_map, mixed_pred]
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

    x_dataset = x_dataset.map(process_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    y_dataset = np.load(LANDAMRKS_PATH)

    y_dataset = tf.data.Dataset.from_tensor_slices(y_dataset)

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    test_dataset = dataset.take(100).batch(BATCH_SIZE)

    valid_dataset = dataset.skip(100).take(100).batch(BATCH_SIZE, drop_remainder=False)

    train_dataset = dataset.skip(200).batch(BATCH_SIZE, drop_remainder=False)

    # train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # print(test_dataset.cardinality(), valid_dataset.cardinality(), train_dataset.cardinality())

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
        # loss = mse(y, preds)
        loss = keras.losses.MSE(y, preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return {'loss':tf.reduce_mean(loss)}



def fit(model, train_dataset, valid_dataset, optim, epoch_counter, ckpt_manager, summary_writer):

    x_sample, y_sample = None, None
    for s in train_dataset.take(1):
        x_sample, y_sample = s

    best_acc = 0.75
    with summary_writer.as_default():
        for e in range(epoch_counter.value(), EPOCH):
            # start = time.time()

            for b, (x_batch, y_batch) in enumerate(train_dataset):
                summary = train_step(model, x_batch, y_batch, optim)

                tf.summary.scalar('loss', summary['loss'], step=e)
                # tf.summary.scalar('lr', learning_rate_fn.current_learning_rate, step=optim.iterations)

            epoch_counter.assign_add(1)

            valid_losses = []
            for b, (x_batch, y_batch) in enumerate(valid_dataset):
                preds = model(x_batch)
                loss = keras.losses.MSE(y_batch, preds)
                valid_losses.append(np.mean(loss.numpy()))

            tf.summary.scalar('valid_loss', np.mean(valid_losses), step=e) 

            if e % 10 == 0:
                save_image(x_sample, y_sample, model(x_sample), e)
                x_valid, y_valid = None, None
                for s in valid_dataset.take(1):
                    x_valid, y_valid = s
                save_image(x_valid, y_valid, model(x_valid), e+1)
                ckpt_manager.save()

            # if np.mean(valid_losses) < best_acc:
            #     best_acc = np.mean(valid_losses)
            #     model.save('bitmoji_lmreg_point')
            #     print('model saved with valid acc : ', best_acc)


if __name__ == '__main__':


    test_dataset, valid_dataset, train_dataset = load_train_dataset()

    model = lmregressor.LandmarkRegressor()

    epoch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # starter_learning_rate = 0.005
    # end_learning_rate = 0.0
    # decay_steps = 200000
    # learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
    #     starter_learning_rate,
    #     decay_steps,
    #     end_learning_rate,
    #     power=2)

    optim = keras.optimizers.Adam(learning_rate=0.0001)

    ckpt = tf.train.Checkpoint(model=model,
                            optim=optim,
                            epoch_counter=epoch_counter)


    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    fit(model, train_dataset, valid_dataset, optim, epoch_counter, ckpt_manager, summary_writer)



    # model = keras.models.load_model('bitmoji_lmreg_point')

    # valid_losses = []
    # for b, (x_batch, y_batch) in enumerate(valid_dataset):
    #     preds = model(x_batch)
    #     loss = keras.losses.MSE(y_batch, preds)
    #     valid_losses.append(np.mean(loss.numpy()))


    # print(valid_losses)
    # print(np.mean(valid_losses))

    # test_losses = []
    # for b, (x_batch, y_batch) in enumerate(test_dataset):
    #     preds = model(x_batch)
    #     loss = keras.losses.MSE(y_batch, preds)
    #     test_losses.append(np.mean(loss.numpy()))


    # print(test_losses)
    # print(np.mean(test_losses))

    # model.save('celeba_lmreg_point')


