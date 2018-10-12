#encoding:utf-8
import cv2
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms

class Drawcnts_and_cut(object):

    def __call__(self, original_img, box):
        filter = lambda x: 0 if x < 0 else x
        Xs = [filter(i[0]) for i in box]
        Ys = [filter(i[1]) for i in box]
        # Xs = [i[0] for i in box]
        # Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img = original_img[y1:y1 + hight, x1:x1 + width]
        return crop_img


class Crop_image(object):

    def __init__(self):
        # set color bound
        self.lower_grey = np.array([0, 0, 46])
        self.upper_grey = np.array([180, 43, 220])
        self.drawcnts_and_cut = Drawcnts_and_cut()

    def __call__(self, image):
        #image = cv2.resize(image, (600, 600))
        
        hsv = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2HSV)
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)

        # cv2.imshow("draw_img", mask)
        # cv2.waitKey(0)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(img_grey, 127, 255, 0)
        # _, contours, hierarcy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        _, contours, hierarcy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return image
        
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        # draw a bounding box arounded the detected barcode and display the image
        # draw_img = cv2.drawContours(image.copy(), [box], -1, (0, 0, 255), 3)
        
        draw_img = image.copy()
        draw_img = cv2.cvtColor(np.asarray(draw_img),cv2.COLOR_RGB2BGR)

        # img_path = img_path.replace(file_path + "\\", "")
        # print(os.path.join(out_file_path, img_path))

        draw_img = Image.fromarray(self.drawcnts_and_cut(draw_img, box))
        
        return draw_img


class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)


def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])


class trainAugment(object):
    def __init__(self):
        self.augment = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        # transforms.RandomRotation(20),
        FixedRotation([0, 90, 180, 270]),
        #transforms.RandomCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def __call__(self, img):
        return self.augment(img)
        

class testAugment(object):
    def __init__(self):
        self.augment = transforms.Compose([
        transforms.Resize((600, 600)),
        #transforms.CenterCrop(384),#384, 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def __call__(self, img):
        return self.augment(img)

