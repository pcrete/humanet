import multiprocessing as mp
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time


def mse(imageA, imageB):
    err = np.sum(np.abs(imageA.astype("float")-imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    err /= 255
    return 1.0-err

def compare_images(img_pairs, LIST):

    for pair  in img_pairs:
        nameA, nameB = pair
        imageA = cv2.imread(os.path.join(directory, nameA))
        imageB = cv2.imread(os.path.join(directory, nameB))

        imageA = cv2.cvtColor(cv2.resize(imageA, (200, 500)), cv2.COLOR_BGR2RGB)
        imageB = cv2.cvtColor(cv2.resize(imageB, (200, 500)), cv2.COLOR_BGR2RGB)
        
        m = mse(imageA, imageB)
        # s = ssim(imageA, imageB, multichannel=True)
        # avg = s*0.6 + m*0.4
        avg = m

        if(avg > 0.85):
            print('Proc ID:', mp.current_process().name, '\tSimilarity (', nameA,',', nameB, ') \t SIM:', avg)
            LIST.append(nameB)

if __name__ == '__main__':

    fig = plt.figure(figsize=(16, 10))

    directory = '../data/3_BEST/output'

    PATH_TO_IMAGES = os.listdir(directory)
    PATH_TO_IMAGES.sort()

    N_IMAGES = len(PATH_TO_IMAGES)
    N_PROC = 8
    procs = []

    print('Total images:', N_IMAGES)

    start = time.time()
    manager = mp.Manager()
    LIST = manager.list()

    image_pairs = []
    for i, imageA in enumerate(PATH_TO_IMAGES):
        for j, imageB in enumerate(PATH_TO_IMAGES[i+1:]):
            image_pairs.append([imageA, imageB])

    N_pairs = len(image_pairs)
    print('Total pairs:', N_pairs)

    batch = int(N_pairs/8)
    print('Batch size:', batch)

    for i in range(N_PROC):
        begin = batch*i
        end = batch*(i+1)

        if( (i+1) == N_PROC and end != N_pairs):
            end = N_pairs

        print('ID:',i,'\t', begin, end-1)
    
        proc = mp.Process(target=compare_images, args=(image_pairs[begin:end], LIST))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    LIST = list(set(LIST))

    print('Total images:', N_IMAGES)
    print('To removes:', len(LIST))

    # for file in LIST:
    #     os.remove(os.path.join(directory, file))

    end = time.time()
    print('Total Time:', str(end-start)[:-8],'seconds')