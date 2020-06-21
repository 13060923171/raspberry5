#!/usr/bin/python
# -*- coding: utf-8 -*-
# 导入工具包
from scipy.spatial import distance as dist
from collections import OrderedDict
import threading as th
import numpy as np
import argparse
import time
import dlib
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (A + B) / (2.0 * C)
    return ear


# 检测与定位工具
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Desktop/test/shape_predictor_68_face_landmarks.dat")

# 分别取两个眼睛区域
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]



def shape_to_np(shape, dtype="int"):
    # 创建68*2
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def camera_thread():
    # 设置判断参数
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3

    # 初始化计数器
    COUNTER = 0
    TOTAL = 0
    print("[INFO] starting video stream thread...")
    camera = PiCamera()
    camera.resolution = (400, 200)
    camera.framerate = 30
    # 将摄像头的视频流转换为数组
    rawCapture = PiRGBArray(camera, size=(400, 200))
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array
        (h, w) = frame.shape[:2]
        width = 1200
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        rects = detector(gray, 0)

        # 遍历每一个检测到的人脸
        for rect in rects:
            # 获取坐标
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # 分别计算ear值
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # 算一个平均的
            ear = (leftEAR + rightEAR) / 2.0

            # 绘制眼睛区域
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # 检查是否满足阈值
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            else:
                # 如果连续几帧都是闭眼的，总数算一次
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # 重置
                COUNTER = 0

            # 显示
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        rawCapture.truncate(0)  # 清除流以准备下一帧
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 相当于time.sleep(1ms)，任意键退出
            camera.close()
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    t1 = th.Thread(target=camera_thread, args=())
    t2 = th.Thread(target=shape_to_np, args=())
    t3 = th.Thread(target=eye_aspect_ratio, args=())

    t3.setDaemon(True)
    t3.start()
    t2.setDaemon(True)
    t2.start()
    t1.setDaemon(True)
    t1.start()

    t3.join()
    t2.join()
    t1.join()

