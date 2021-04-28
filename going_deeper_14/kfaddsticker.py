import cv2
import numpy as np
import dlib
from tqdm import tqdm

from kpkf import tracker

detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

box_tracker = tracker(num_points=2, sys_err=0.5, measure_err=1000)
nose_tracker = tracker(num_points=1, sys_err=1., measure_err=100)

flg_nose_tracker = True

def draw_mid(img_bgr, list_landmarks, list_bbox):
    for bbox in list_bbox:
        l = bbox.left()
        t = bbox.top()
        r = bbox.right()
        b = bbox.bottom()
        l,t,r,b = [ele*2 for ele in [l,t,r,b]]
        img_bgr = cv2.rectangle(img_bgr, (l,t), (r,b), (0,255,0), 2, cv2.LINE_AA)
    for lm in list_landmarks:
        for pt in lm:
            pt = tuple([ele*2 for ele in pt])
            img_bgr = cv2.circle(img_bgr, pt, 1, (0,128,255), -1, cv2.LINE_AA)
    return img_bgr

def img2sticker_kf(img_orig, img_sticker, detector_hog, landmark_predictor):
    # preprocess
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # detector
    img_rgb_vga = cv2.resize(img_rgb, (640, 360))
    dlib_rects = detector_hog(img_rgb_vga, 0)
    if len(dlib_rects) < 1:
        return img_orig
    
    # tracker
    if len(dlib_rects) == 1:
        bbox = dlib_rects[0]
        list_input = [(bbox.left(), bbox.top()), (bbox.right(), bbox.bottom())]
        np_estimate = np.array(box_tracker.process(list_input))
        np_est_points = np_estimate.reshape(2, 3)[:,:2].astype(int)
        l,t,r,b = np_est_points.flatten()
        # print (l,t,r,b)
        if (b-t)*(r-l) > 100:
            dlib_rects[0] = dlib.rectangle(left=l,top=t,right=r,bottom=b)

    # landmark
    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb_vga, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)
    
    # head coord
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x = landmark[30][0] # nose
        y = landmark[30][1] 
        w = dlib_rect.width()
        h = dlib_rect.width()
        x,y = [ele*2 for ele in [x,y]]
        break
    # sticker
    img_sticker = cv2.resize(img_sticker, (w,h), interpolation=cv2.INTER_NEAREST)
    
    if flg_nose_tracker == True:
        list_input = [(x, y)]
        np_estimate = np.array(nose_tracker.process(list_input))
        np_est_points = np_estimate.reshape(1, 3)[:,:2].astype(int)
        x_tmp, y_tmp = np_est_points.flatten()
        if x_tmp*y_tmp != 0:
            x = x_tmp
            y = y_tmp

    refined_x = x - w // 2
    refined_y = y - h - dlib_rect.width()
    # print ('(x,y) : (%d,%d)'%(refined_x, refined_y))

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:]
        refined_y = 0

    if refined_x < 0:
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0
    elif refined_x + img_sticker.shape[1] >= img_orig.shape[1]:
        img_sticker = img_sticker[:, :-(img_sticker.shape[1]+refined_x-img_orig.shape[1])]


    img_bgr = img_orig.copy()
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        cv2.addWeighted(sticker_area, 1.0, img_sticker, 0.7, 0)

    return img_bgr
