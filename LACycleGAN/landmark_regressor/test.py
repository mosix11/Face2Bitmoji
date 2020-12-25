import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import cv2
import random

DATA_DIR = '../celeba2bitmoji/trainB/'
HEATMAP_DIR = '../celeba2bitmoji/trainB_lmheatmap/'



def get_sorted_file_names(base_dir):
    paths = os.listdir(base_dir)
    paths = [int(p.replace('.jpg', '')) for p in paths]
    paths = sorted(paths)
    paths = [str(p)+'.jpg' for p in paths]
    for i, img_name in enumerate(paths):
        if len(img_name) < 8:
            difference = 8 - len(img_name)
            img_name = difference * '0' + img_name
        paths[i] = img_name

    return paths

def normalize(img, current_min, current_max, desired_min, desired_max):
    img = (img - current_min) * ((desired_max - desired_min) / (current_max - current_min)) + desired_min
    return img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def show_image(x, y, preds, code):
    x = x[0]
    y = y[0]
    pred = preds[0].numpy()
    pred = normalize(pred, np.min(pred), np.max(pred), 0, 1)
    plt.figure(figsize=(15, 15))

    flipped = cv2.flip(x, 1)
    # flipped = rotate_image(flipped, 25)

    mixed_pred = np.clip(flipped + pred, 0, 1)
    mixed_trgt = np.clip(x + y, 0, 1)
    display_list = [x, y, mixed_trgt, pred, mixed_pred]
    title = ['Input Image', 'Target Image', 'Target Mixed Image', 'Predicted Image', 'Pred Mixed Image']

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def show_image_2(x, y, code):
    x = normalize(x, 0, 255, 0, 1)
    y = normalize(y, 0, 255, 0, 1)
    mixed = np.zeros_like(x)
    mixed = x + y
    mixed_trgt = np.clip(mixed, 0, 1)
    # mixed_trgt = normalize(mixed_trgt, 0, 255, 0, 1)
    display_list = [x, y, mixed_trgt]
    # title = ['Input Image', 'Target Image', 'Mixed Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        # plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    filename = 'img'+str(code)+'.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def show_image_3(x, y, preds, code):
    x = x[0]
    y = y[0]
    pred = preds[0].numpy()
    pred = normalize(pred, np.min(pred), np.max(pred), 0, 1)
    plt.figure(figsize=(15, 15))

    flipped = cv2.flip(x, 1)
    # flipped = rotate_image(flipped, 25)

    mixed_pred = np.clip(flipped + pred, 0, 1)
    mixed_trgt = np.clip(x + y, 0, 1)
    display_list = [x, y, mixed_trgt, pred, mixed_pred]

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(display_list[i])
        plt.axis('off')
    filename = 'img'+str(code)+'.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
   



model = keras.models.load_model('bitmoji_lmreg_heat')

# model.save('celeba_lmreg_heat.h5')


paths_img = get_sorted_file_names(DATA_DIR)
paths_lm = get_sorted_file_names(HEATMAP_DIR)



# img_path = paths_img[18]
# lm_path = paths_lm[18]
# img = cv2.imread(DATA_DIR + img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# landmark = cv2.imread(HEATMAP_DIR + lm_path)
# landmark = cv2.cvtColor(landmark, cv2.COLOR_BGR2RGB)
# show_image_2(img, landmark, 18)

cnt = 0
for i in np.random.randint(0, 4000, 100):
    img_path = paths_img[i]
    lm_path = paths_lm[i]

    img = cv2.imread(DATA_DIR + img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis]
    img = img.astype(np.float)
    img = normalize(img, 0, 255, 0, 1)

    landmark = cv2.imread(HEATMAP_DIR + lm_path)
    landmark = cv2.cvtColor(landmark, cv2.COLOR_BGR2RGB)
    landmark = landmark[np.newaxis]
    landmark = landmark.astype(np.float)
    landmark = normalize(landmark, 0, 255, 0, 1)

    flipped = cv2.flip(img[0], 1)
    # flipped = rotate_image(flipped, 25)
    flipped = flipped[np.newaxis]

    preds = model(flipped)
    show_image_3(img, landmark, preds, i)

    cnt += 1
    if cnt == 6:
        break

