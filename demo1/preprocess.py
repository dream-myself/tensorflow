import cv2
import numpy as np
images = ['deal/2.txt']
def read_data(paths):
    image_list = []
    label_list = []
    for path in paths:

        with open(path, 'r') as f:
            line = f.readline()

            while line:
                example = line.rstrip().split('--')
                image = cv2.imread(example[0], cv2.IMREAD_GRAYSCALE)
                image_list.append(image.flatten() / 255)
                label_list.append(deal_label(example[1]))
                line = f.readline()

    return np.array(image_list), np.array(label_list)



def deal_label(label):
    num_letter = dict(enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
    letter_num = dict(zip(num_letter.values(), num_letter.keys()))
    letter_list = []
    try:
        for i in label:
            letter_list.append(letter_num[i])
    except Exception as ex:
        print(label)

    return np.array(letter_list)


if __name__ == "__main__":
    read_data(images)