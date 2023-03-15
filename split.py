import cv2 as cv
import numpy as np
import os
import json

raw_path = r'/home/boyd/PycharmProjects/MarsSurface/raw/'
x = []
y = []
relative_point = []
class_num = {'bedrock': 0, 'largerock': 0, 'sky': 0, 'void': 0, 'sand': 0, 'rover': 0, 'mountain': 0, 'rockregion': 0,'wheel':0}
raw_list = os.listdir(raw_path)
print(raw_list)
for raw_filename in raw_list:

    name, file_type = raw_filename.split('.')
    if file_type == 'json':
        continue
    srcimg = cv.imread(raw_path + raw_filename)
    with open(raw_path + name + '.json', 'r') as j:
        j_data = json.load(j)
        shapes = j_data['shapes']
        for shape in shapes:
            label = shape['label']
            if label=='mountain':
                print(label)
            points = shape['points']
            for point in points:
                x.append(point[0])
                y.append(point[1])
                max_x = int(max(x))
                min_x = int(min(x))
                max_y = int(max(y))
                min_y = int(min(y))
            for point in points:
                relative_point.append([point[0] - min_x, point[1] - min_y])
            min_rect_img = srcimg[min_y:max_y, min_x:max_x]
            mask = np.zeros(min_rect_img.shape, min_rect_img.dtype)
            cv.fillPoly(mask, [np.array(relative_point, dtype=np.int32)], (255, 255, 255))
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            res = cv.bitwise_and(min_rect_img, min_rect_img, mask=mask)
            cv.imwrite('dataset-full/' + label + '/' + label + '-' + str(class_num[label]) + '.png', res)
            relative_dict = {'label': label, 'points': relative_point}
            with open('dataset-full/' + label + '/' + label + '-' + str(class_num[label]) + '.json', 'w') as fs:
                json.dump(relative_dict, fs)
                relative_dict.clear()
            x.clear()
            y.clear()
            class_num[label] += 1

            relative_point.clear()
            print(class_num['mountain'])
