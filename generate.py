import json
import numpy as np
import cv2 as cv
import os
import random


def enhancement(img, points):
    e_type = random.randint(1, 3)
    cj_type = ['b', 's', 'c']
    noise_type = ['gauss', 'sp']
    f_type = ["blur", 'gaussian', 'median']
    img, points = random_resize(img, points)
    if e_type == 1:
        img = colorjitter(img, random.choice(cj_type))
    elif e_type == 2:
        img = noise(img, random.choice(noise_type))
    elif e_type == 3:
        img = filters(img, random.choice(f_type))
    return img, points


def random_resize(img, points):
    resize_ratio = random.uniform(0.8, 1.2)
    h, w = img.shape[:2]
    size = (int(w * resize_ratio), int(h * resize_ratio))
    img = cv.resize(img, size)
    for point in points:
        point[0] *= resize_ratio
        point[1] *= resize_ratio
    return img, points


def colorjitter(img, cj_type="b"):
    """
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: contrast}
    """
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv.merge((h, s, v))
        img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv.merge((h, s, v))
        img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img


def noise(img, noise_type="gauss"):
    """
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    """
    if noise_type == "gauss":
        image = img.copy()
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        image = cv.add(image, gauss)
        return image

    elif noise_type == "sp":
        image = img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image


def filters(img, f_type="blur"):
    """
    ### Filtering ###
    img: image
    f_type: {blur: blur, gaussian: gaussian, median: median}
    """
    if f_type == "blur":
        image = img.copy()
        fsize = 3
        return cv.blur(image, (fsize, fsize))

    elif f_type == "gaussian":
        image = img.copy()
        fsize = 3
        return cv.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image = img.copy()
        fsize = 3
        return cv.medianBlur(image, fsize)


def is_overlap(pre_boxes, bbox):
    for pre_box in pre_boxes:
        if ((pre_box[1][1] > bbox[0][1]) and (pre_box[0][1] < bbox[1][1]) and (pre_box[1][0] > bbox[0][0]) and (
                pre_box[0][0] < bbox[1][0])):
            return False
    return True


def get_rate(bg_w, bg_h, fg_w, fg_h):
    rate1 = 0
    rate2 = 0
    rate = 1
    if fg_w > bg_w:
        rate1 = fg_w / bg_w
    if fg_h > bg_h:
        rate2 = fg_h / bg_w
    if rate1 == 0 and rate2 == 0:
        return rate
    else:
        return (1 / max(rate1, rate2)) - 0.1


def main():
    ############################################################

    class_num = {'sand': 1, 'rover': 0, 'mountain': 1, 'rockregion': 1, 'wheel': 0,  'bedrock': 5,
                 'largerock': 5, 'sky': 0}
    loc_limit = {'bedrock': [[0, 0.6], [1, 1]], 'largerock': [[0, 0.6], [1, 1]], 'sky': [], 'sand': [[0, 0.2], [1, 1]],
                 'rover': [],
                 'mountain': [[0, 0.1], [1, 0.4]], 'rockregion': [[0, 0.2], [1, 1]],
                 'wheel': [[0, 0], [1, 1]]}  # [start(x,y),end(x,y)]
    mode = 'demo'
    if mode == 'full':
        dataset_dir = 'dataset-demo/'
        save_dir = 'res-demo/'
    else:
        dataset_dir = 'dataset-full/'
        save_dir = 'res-full/'
    total_num = 10

    ############################################################

    bg_dir = dataset_dir + 'backgrounds/'
    bg_list = os.listdir(bg_dir)
    pre_boxes = []
    x = []
    y = []
    tmp_x = []
    tmp_y = []
    for bg_num in range(0, total_num):
        bg = cv.imread(bg_dir + random.choice(bg_list))
        bg=cv.resize(bg,(2560,1600))
        bg_h, bg_w = bg.shape[:2]
        bg_dict = {'shapes': []}
        first_flag = False
        for cls in class_num:
            cls_list = os.listdir(dataset_dir + cls + '/')
            for i in cls_list:
                if i.endswith('.json'):
                    cls_list.remove(i)
                    
            for i in range(0, class_num[cls]):
                # print(cls, i)
                fg_filename = dataset_dir + cls + '/' + random.choice(cls_list)
                fg = cv.imread(fg_filename)
                with open(fg_filename[:-3] + 'json', 'r') as j:
                    j_data = json.load(j)
                    fg_lable = j_data['label']
                    fg_points = j_data['points']

                fg, fg_points = enhancement(fg, fg_points)
                fg_h, fg_w = fg.shape[:2]
                mask = np.zeros(fg.shape, fg.dtype)
                cv.fillPoly(mask, [np.array(fg_points, dtype=np.int32)], (255, 255, 255))
                mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
                # cv.imshow(fg_filename,fg)
                # cv.imshow(fg_filename+'mask',mask)
                # cv.waitKey(0)
                rate = get_rate(bg_w, bg_h, fg_w, fg_h)
                size = (int(fg_w * rate), int(fg_h * rate))
                fg = cv.resize(fg, size)
                mask = cv.resize(mask, size)
                fg_w = int(fg_w * rate)
                fg_h = int(fg_h * rate)
                for fg_point in fg_points:
                    fg_point[0] *= rate
                    fg_point[1] *= rate
                if not first_flag:
                    offset_x = random.randint(int(bg_w * loc_limit[cls][0][0]),
                                              int((bg_w - fg_w) * loc_limit[cls][1][0]))
                    offset_y = random.randint(int(bg_h * loc_limit[cls][0][1]),
                                              int((bg_h - fg_h) * loc_limit[cls][1][1]))
                    for fg_point in fg_points:
                        fg_point[0] += offset_x
                        fg_point[1] += offset_y
                        x.append(fg_point[0])
                        y.append(fg_point[1])
                        max_x = int(max(x))
                        min_x = int(min(x))
                        max_y = int(max(y))
                        min_y = int(min(y))
                    pre_boxes.append([[min_x, min_y], [max_x, max_y]])
                    first_flag = True
                else:
                    tmp_x.clear()
                    tmp_y.clear()
                    offset_x = random.randint(int(bg_w * loc_limit[cls][0][0]),
                                              int((bg_w - fg_w) * loc_limit[cls][1][0]))
                    offset_y = random.randint(int(bg_h * loc_limit[cls][0][1]),
                                              int((bg_h - fg_h) * loc_limit[cls][1][1]))
                    for fg_point in fg_points:
                        tmp_x.append(fg_point[0] + offset_x)
                        tmp_y.append(fg_point[1] + offset_y)
                        max_x = int(max(tmp_x))
                        min_x = int(min(tmp_x))
                        max_y = int(max(tmp_y))
                        min_y = int(min(tmp_y))
                    while not is_overlap(pre_boxes, [[min_x, min_y], [max_x, max_y]]):
                        tmp_x.clear()
                        tmp_y.clear()
                        offset_x = random.randint(int(bg_w * loc_limit[cls][0][0]),
                                                  int((bg_w - fg_w) * loc_limit[cls][1][0]))
                        offset_y = random.randint(int(bg_h * loc_limit[cls][0][1]),
                                                  int((bg_h - fg_h) * loc_limit[cls][1][1]))
                        for fg_point in fg_points:
                            tmp_x.append(fg_point[0] + offset_x)
                            tmp_y.append(fg_point[1] + offset_y)
                            max_x = int(max(tmp_x))
                            min_x = int(min(tmp_x))
                            max_y = int(max(tmp_y))
                            min_y = int(min(tmp_y))
                    for i in range(0, len(fg_points)):
                        fg_points[i] = [tmp_x[i], tmp_y[i]]

                    pre_boxes.append([[min_x, min_y], [max_x, max_y]])

                # merge
                mask_inv = cv.bitwise_not(mask)
                bg[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w] = cv.bitwise_and(
                    bg[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w],
                    bg[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w], mask=mask_inv)
                tmp = cv.bitwise_and(fg, fg, mask=mask)
                bg[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w] = cv.add(
                    bg[offset_y:offset_y + fg_h, offset_x:offset_x + fg_w], tmp)
                bg_dict['shapes'].append({'label': fg_lable, 'points': fg_points})
                x.clear()
                y.clear()
                # cv.namedWindow('a', cv.WINDOW_NORMAL + cv.WINDOW_KEEPRATIO)
                # cv.imshow('a', bg)
                # cv.waitKey(0)
        # save each pic
        pre_boxes.clear()
        with open(save_dir + 'res-' + str(bg_num) + '.json', 'w') as js:
            json.dump(bg_dict, js)
        cv.imwrite(save_dir + 'res-' + str(bg_num) + '.png', bg)
        print('saved')


if __name__ == '__main__':
    main()
