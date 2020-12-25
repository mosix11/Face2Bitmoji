import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.integrate as inetgrate

BASE_DIR = '../celeba2bitmoji/trainB/'
SAVE_DIR = '../celeba2bitmoji/trainB_lmheatmap/'
LANDMARKS_PATH = '../celeba2bitmoji/bitmoji_landmarks.npy'



def get_sorted_file_names():
    paths = os.listdir(BASE_DIR)
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


def show_img(img, land):

    heatmap = np.zeros_like(img)
    color = (255,255,255)
    cv2.circle(heatmap, (land[0], land[1]), 2, color, 2)
    cv2.circle(heatmap, (land[2], land[3]), 2, color, 2)
    cv2.circle(heatmap, (land[4], land[5]), 2, color, 2)
    cv2.circle(heatmap, (land[6], land[7]), 2, color, 2)
    cv2.circle(heatmap, (land[8], land[9]), 2, color, 2)

    img = normalize(img, 0, 255, 0, 1)
    heatmap = normalize(heatmap, 0, 255, 0, 1)
    # mixed = np.clip(img + heatmap, 0, 1)

    plt.imshow(heatmap)
    plt.show()


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
    
    heat_elem = normalize(heat_elem, 0.0, np.max(heat_elem), 0.0, 255.0).astype(np.int)
    
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



file_names = get_sorted_file_names()
landmarks = np.load(LANDMARKS_PATH)

# for i, file_name in enumerate(file_names):
#     img = cv2.imread(BASE_DIR + file_name)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     heatmap = cv2.imread(SAVE_DIR + file_name)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     mixed = np.clip(img.astype(float) + heatmap.astype(float), 0, 255)
#     mixed = normalize(mixed, 0, 255, 0, 1)
#     plt.imshow(mixed)
#     plt.show()

# for i, file_name in enumerate(file_names):
#     img = cv2.imread(BASE_DIR + file_name)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     heatmap = generate_heatmap(img, landmarks[i])
    
#     # print(np.min(heatmap), np.max(heatmap))
#     # plt.imshow(heatmap)
#     # plt.show()

#     cv2.imwrite(SAVE_DIR + file_name, heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



# show_img(normalize(img[0], 0, 255, 0, 1), lands)