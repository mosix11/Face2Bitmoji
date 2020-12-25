import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import data.dataset as ds
import data.image_util as im_util
import model.modules as modules
import model.networks as networks
import config
import time
import datetime
import cv2

# starter_learning_rate = 0.0002
# end_learning_rate = 0
# decay_steps = 100000
# power = 4

# def decayed_learning_rate(step):
#   step = min(step, decay_steps)
#   return ((starter_learning_rate - end_learning_rate) * (1 - step / decay_steps)**(power)) + end_learning_rate

# x = np.linspace(1, 100000, 100000, dtype=np.int)

# y = [decayed_learning_rate(i) for i in x]
# y = np.asarray(y)

# plt.plot(x, y)
# plt.show()






def test_model(code):
    A_imgs = []
    A_lms= []
    B_imgs = []
    B_lms = []
    for img in A_set:
        A_imgs.append(img[0][0].numpy())
        A_lms.append(img[1][0].numpy())
    for img in B_set:
        B_imgs.append(img[0][0].numpy())
        B_lms.append(img[1][0].numpy())



    A_imgs = np.asarray(A_imgs)
    A_lms = np.asarray(A_lms)
    B_imgs = np.asarray(B_imgs)
    B_lms= np.asarray(B_lms)

    print(A_imgs.shape)

    A2B_preds = gen_A2B([A_imgs, A_lms], training=True)
    B2A_preds = gen_B2A([B_imgs, B_lms], training=True)

    A2B_lms = B_lm_reg(A2B_preds)
    B2A_lms = A_lm_reg(B2A_preds)

    A_restored = gen_B2A([A2B_preds, A2B_lms], training=True)
    B_restored = gen_A2B([B2A_preds, B2A_lms], training=True)

    A_imgs = (A_imgs + 1) / 2
    B_imgs = (B_imgs + 1) / 2
    A2B_preds = (A2B_preds + 1) / 2
    B2A_preds = (B2A_preds + 1) / 2
    A_restored = (A_restored + 1) / 2
    B_restored = (B_restored + 1) / 2

    # titles = ['A Image', 'Predicted B Image', 'A Restored', 'B Image', 'Predicted B Image', 'B Restored']

    fig, axes = plt.subplots(test_samples,3)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    axes[0].imshow(A_imgs[0])
    axes[0].axis(False)
    axes[1].imshow(A2B_preds[0])
    axes[1].axis(False)
    axes[2].imshow(A_restored[0])
    axes[2].axis(False)
    # for i in range(test_samples):
    #     for j in range(3):
    #         if i == 0 :
    #             # plt.title(titles[j])
    #             pass
    #         if j == 0:
    #             axes[i,j].imshow(A_imgs[i])
    #             axes[i,j].axis(False)
    #         elif j == 1 :
    #             axes[i,j].imshow(A2B_preds[i])
    #             axes[i,j].axis(False)
    #         elif j == 2 :
    #             axes[i,j].imshow(A_restored[i])
    #             axes[i,j].axis(False)
            # elif j == 3 :
            #     axes[i,j].imshow(B_imgs[i])
            #     axes[i,j].axis(False)
            # elif j == 4 :
            #     axes[i,j].imshow(B2A_preds[i])
            #     axes[i,j].axis(False)
            # elif j == 5 :
            #     axes[i,j].imshow(B_restored[i])
            #     axes[i,j].axis(False)
    # plt.show()
    filename = 'img'+str(code)+'.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def gauss_map(size_x, size_y=None, sigma_x=2.3, sigma_y=None):    
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x
    
    assert isinstance(size_x, int)
    assert isinstance(size_y, int)
    
    x0 = size_x // 2
    y0 = size_y // 2
    
    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:,np.newaxis]
    
    x -= x0
    y -= y0
    
    exp_part = x**2/(2*sigma_x**2)+ y**2/(2*sigma_y**2)
    return 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-exp_part)

def generate_heatmap(img, lands):
    heatmap = np.zeros_like(img, dtype=np.int)

    elem_size = 15
    elem_rad = int(elem_size/2)
    heat_elem = gauss_map(elem_size)
    
    heat_elem = im_util.normalize(heat_elem, 0.0, np.max(heat_elem), 0.0, 255.0).astype(np.int)
    
    lands_center = []
    for p in range(0, 10, 2):
        lands_center.append((lands[p+1], lands[p])) 

    for c_x, c_y in lands_center:
        start_idx = (c_x - elem_rad, c_y - elem_rad)
        end_idx = (c_x + elem_rad, c_y + elem_rad)

        heatmap[start_idx[0]:end_idx[0]+1, start_idx[1]:end_idx[1]+1, 0] = heat_elem
        heatmap[start_idx[0]:end_idx[0]+1, start_idx[1]:end_idx[1]+1, 1] = heat_elem
        heatmap[start_idx[0]:end_idx[0]+1, start_idx[1]:end_idx[1]+1, 2] = heat_elem

    return heatmap


disc_UGD_A = networks.UnconditionalGlobalDiscriminator()
disc_CGD_A = networks.ConditionalGlobalDiscriminator()
gen_B2A = networks.Generator()
A_lm_reg = keras.models.load_model(config.A_LANDMARK_REGRESSOR_PATH)


disc_UGD_B = networks.UnconditionalGlobalDiscriminator()
disc_CGD_B = networks.ConditionalGlobalDiscriminator()
gen_A2B = networks.Generator()
B_lm_reg = keras.models.load_model(config.B_LANDMARK_REGRESSOR_PATH)


gen_lr_scheduler = keras.optimizers.schedules.PolynomialDecay(config.START_LR, config.ITERS, config.END_LR, power=config.DECAY_POWER)

disc_UGD_A_lr_scheduler = keras.optimizers.schedules.PolynomialDecay(config.START_LR, config.ITERS, config.END_LR, power=config.DECAY_POWER)
disc_CGD_A_lr_scheduler = keras.optimizers.schedules.PolynomialDecay(config.START_LR, config.ITERS, config.END_LR, power=config.DECAY_POWER)

disc_UGD_B_lr_scheduler = keras.optimizers.schedules.PolynomialDecay(config.START_LR, config.ITERS, config.END_LR, power=config.DECAY_POWER)
disc_CGD_B_lr_scheduler = keras.optimizers.schedules.PolynomialDecay(config.START_LR, config.ITERS, config.END_LR, power=config.DECAY_POWER)

# gen_lr_scheduler = modules.LinearDecay(2e-4, config.ITERS, config.ITERS_DECAY_START)

# disc_UGD_A_lr_scheduler = modules.LinearDecay(2e-4, config.ITERS, config.ITERS_DECAY_START)
# disc_CGD_A_lr_scheduler = modules.LinearDecay(2e-4, config.ITERS, config.ITERS_DECAY_START)

# disc_UGD_B_lr_scheduler = modules.LinearDecay(2e-4, config.ITERS, config.ITERS_DECAY_START)
# disc_CGD_B_lr_scheduler = modules.LinearDecay(2e-4, config.ITERS, config.ITERS_DECAY_START)


gen_optimizer = keras.optimizers.Adam(gen_lr_scheduler, beta_1=0.5)


disc_UGD_A_optimizer = keras.optimizers.Adam(disc_UGD_A_lr_scheduler, beta_1=0.5)
disc_CGD_A_optimizer = keras.optimizers.Adam(disc_CGD_A_lr_scheduler, beta_1=0.5)

disc_UGD_B_optimizer = keras.optimizers.Adam(disc_UGD_B_lr_scheduler, beta_1=0.5)
disc_CGD_B_optimizer = keras.optimizers.Adam(disc_CGD_B_lr_scheduler, beta_1=0.5)


iter_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

ckpt = tf.train.Checkpoint(gen_A2B=gen_A2B,
                        gen_B2A=gen_B2A,
                        disc_UGD_A=disc_UGD_A,
                        disc_CGD_A=disc_CGD_A,
                        disc_UGD_B=disc_UGD_B,
                        disc_CGD_B=disc_CGD_B,
                        gen_optimizer=gen_optimizer,    
                        disc_UGD_A_optimizer=disc_UGD_A_optimizer,
                        disc_CGD_A_optimizer=disc_CGD_A_optimizer,
                        disc_UGD_B_optimizer=disc_UGD_B_optimizer,
                        disc_CGD_B_optimizer=disc_CGD_B_optimizer,
                        iter_counter=iter_counter)


ckpt_manager = tf.train.CheckpointManager(ckpt, config.CHECKPOINT_PATH, max_to_keep=20)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(config.CHECKPOINT_PATH + '/ckpt-79')
    print ('Latest checkpoint restored!!')

# for s in range(20):
#     test_samples = 1

#     A_set, B_set = ds.load_train_dataset()
#     A_set, B_set = A_set.take(test_samples), B_set.take(test_samples)

#     test_model(s*10)

img = cv2.imread('myimg.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
landmarks = np.array([48, 70, 78, 68, 64, 85, 51, 100, 77, 99])

heatmap = generate_heatmap(img, landmarks)

# img = tf.data.Dataset.from_tensor_slices(img)
# landmarks = tf.data.Dataset.from_tensor_slices(landmarks)
# print(img.take(1))
# print(landmarks.take(1))
# plt.imshow(img)
# plt.show()

img = img.astype(np.float)
img = im_util.normalize(img, 0, 255, -1, 1)
img = img[np.newaxis, ...]
heatmap = heatmap.astype(np.float)
heatmap = im_util.normalize(heatmap, 0, 255, -1, 1)
heatmap = heatmap[np.newaxis, ...]


# mix = np.clip((img+heatmap), 0, 1)
# plt.imshow(mix)
# plt.show()




A_imgs = img
A_lms = heatmap

A2B_preds = gen_A2B([A_imgs, A_lms], training=True)

A2B_lms = B_lm_reg(A2B_preds)

A_restored = gen_B2A([A2B_preds, A2B_lms], training=True)

A_imgs = (A_imgs + 1) / 2
A2B_preds = (A2B_preds + 1) / 2
A_restored = (A_restored + 1) / 2

fig, axes = plt.subplots(1,3)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
axes[0].imshow(A_imgs[0])
axes[0].axis(False)
axes[1].imshow(A2B_preds[0])
axes[1].axis(False)
axes[2].imshow(A_restored[0])
axes[2].axis(False)

filename = 'img'+str(90)+'.png'
plt.savefig(filename, bbox_inches='tight')
plt.close()