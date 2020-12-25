import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import face_detection
import mtcnn

BASE_DIR = ''
BATCH_SIZE = 1

RetinaFace = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
mtcnn_model = mtcnn.MTCNN()

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

    mixed = np.clip(img + heatmap, 0, 1)

    plt.imshow(mixed)
    plt.show()
    

def get_sorted_file_names():
    paths = os.listdir(BASE_DIR)
    paths = [int(p.replace('.jpg', '')) for p in paths]
    paths = sorted(paths)
    paths = [str(p)+'.jpg' for p in paths]
    for i, img_name in enumerate(paths):
        if len(img_name) < 10:
            difference = 10 - len(img_name)
            img_name = difference * '0' + img_name
        paths[i] = img_name

    return paths

paths = get_sorted_file_names()
num_files = len(paths)

# saved_landmarks = np.load('bitmoji_landmarks.npy')
# print(saved_landmarks.shape)

landmarks = np.empty((num_files, 10), dtype=np.int)
deletes = []
for i,path in enumerate(paths):
    img = cv2.imread(BASE_DIR + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    _, lands = RetinaFace.batched_detect_with_landmarks(img)

    lands = np.asarray(lands)
    lands = np.rint(lands)
    lands = lands.astype(int)

    if lands.shape != (1,1,5,2):
        mtcnn_dets = mtcnn_model.detect_faces(img[0])
        if len(mtcnn_dets) == 0:
            print(path, ' REMOVED')
            # os.remove(BASE_DIR + path)
            deletes.append(i)
            continue
        elif len(mtcnn_dets) > 1:
            print('More than 1 found for :', path)
        else:
            keypoints = mtcnn_dets[0]['keypoints']
            lands = np.zeros((5,2), np.int)
            lands[0] = keypoints['left_eye']
            lands[1] = keypoints['right_eye']
            lands[2] = keypoints['nose']
            lands[3] = keypoints['mouth_left']
            lands[4] = keypoints['mouth_right']



    lands = lands.flatten()

    # if not np.array_equal(lands, saved_landmarks[i]):
    #     print(path)
    #     print(landmarks[i])
    #     print(saved_landmarks[i])

    landmarks[i] = lands




    # show_img(normalize(img[0], 0, 255, 0, 1), lands)
    # if i == 20:
    #     break
    if i % 2000 == 0:
        print('Passsed ', i)

# print(len(deletes))

# print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))

# landmarks = np.delete(landmarks, deletes, axis=0)
# print('maxxxxx : ', np.max(landmarks), '    minnnnnnnnnnnnnnnnnnn : ', np.min(landmarks))

# np.save('bitmoji_landmarks.npy', landmarks)





