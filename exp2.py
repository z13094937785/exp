import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import threading
import time
import cv2
import HandTrackingModule as htm
import autopy
import numpy as np
import whisper
import zhconv
import wave
import pyaudio
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from socket import *

def hand():
    wCam, hCam = 1080, 720
    frameR = 100
    smoothening = 5

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    detector = htm.handDetector()
    wScr, hScr = autopy.screen.size()
    # print(wScr, hScr)
    a, b, c, d, e, f = [], [], [], [], [], []
    global old, smoothed_signal, record
    record = [a, b, c, d, e, f]
    ii = 0
    si = 0
    s = []
    sigma = 5
    size = 5
    global julu, start, julu_start, chongfu_start
    start = False
    time1 = time.time()
    out_signal = [0, 0, 0, 0, 0, 0]
    jilu = []
    shunxu = []
    jilu_index = 0
    old = [0, 0, 0, 0, 0, 0]
    chongfu_index = 0

    global msg
    msg = [0, 0, 0, 0, 0, 0]

    def gaussian(signal, sigma, size):
        center_index = size // 2
        x = np.arange(size) - center_index
        kernel = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        smoothed_signal1 = np.convolve(signal[0][-9:], kernel, mode='same')
        smoothed_signal2 = np.convolve(signal[1][-9:], kernel, mode='same')
        smoothed_signal3 = np.convolve(signal[2][-9:], kernel, mode='same')
        smoothed_signal4 = np.convolve(signal[3][-9:], kernel, mode='same')
        smoothed_signal5 = np.convolve(signal[4][-9:], kernel, mode='same')
        smoothed_signal6 = np.convolve(signal[5][-9:], kernel, mode='same')

        smoothed_signal = [int(smoothed_signal1[4]), int(smoothed_signal2[4]), int(smoothed_signal3[4]),
                           int(smoothed_signal4[4]), int(smoothed_signal5[4]), int(smoothed_signal6[4])]

        return smoothed_signal
while True:
    time0 = time.time()
    _, img = cap.read()
    img = detector.findHands(img)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2, cv2.FONT_HERSHEY_PLAIN)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1 = lmList[0][1]

        x21, y21 = lmList[16][1:]
        x22, y22 = lmList[13][1:]

        x31, y31 = lmList[12][1:]
        x32, y32 = lmList[9][1:]

        x41, y41 = lmList[8][1:]
        x42, y42 = lmList[5][1:]

        try:
            x51, y51 = lmList[30][1:]
            x52, y52 = lmList[21][1:]

            x61, y61 = lmList[25][1:]
            x62, y62 = lmList[29][1:]

            x71, y71 = lmList[33][1:]
            x72, y72 = lmList[30][1:]

            x81, y81 = lmList[37][1:]
            x82, y82 = lmList[34][1:]

        except:
            x51, y51 = 0, 0
            x52, y52 = 0, 0

            x61, y61 = 0, 0
            x62, y62 = 0, 0

            x71, y71 = 0, 0
            x72, y72 = 0, 0

            x81, y81 = 0, 0
            x82, y82 = 0, 0

        angle1 = np.around((np.maximum(x1 - 650, 0) / 500 * 120), 2)
        angle2 = np.around((np.maximum(math.hypot(x21 - x22, y21 - y22) - 70, 0)) / 100 * 100, 2)
        angle3 = np.around(np.maximum(math.hypot(x31 - x32, y31 - y32) - 90, 0), 2)
        angle4 = np.around(np.maximum(math.hypot(x41 - x42, y41 - y42) - 85, 0), 2)
        angle5 = np.around(np.maximum(math.atan2(y52 - y51, x52 - x51) * 180 / math.pi - 65, 0), 2)
        angle6 = np.around(np.maximum(math.hypot(x61 - x62, y61 - y62) - 50, 0) / 200 * 90, 2)

        zhongzhi = np.around(np.maximum(math.hypot(x71 - x72, y71 - y72), 0), 2)
        wuming = np.around(np.maximum(math.hypot(x81 - x82, y81 - y82), 0), 2)
        if 0 < zhongzhi < 120 and time0 - time1 > 1.5:
            start = not start
            time1 = time.time()  # 0
            print(start)
        out = [angle1, angle2, angle3, angle4, angle5, angle6]

        for i in range(6):
            if abs(out[i] - old[i]) > 1:
                out_signal[i] = out[i]
            else:
                out_signal[i] = old[i]
        # print(out)
        for i in range(6):
            record[i].append(out_signal[i])
        # print(record)
        ii += 1
        if start and si < 20:
            s.append(record[2][-1])

            if si == 19:
                leiji = 0
                pingjun = np.sum(s) / len(s)
                for i in range(len(s)):
                    leiji += (s[i] - pingjun) ** 2

                sigma = np.sqrt(leiji)
                print(sigma)
            si += 1
        if sigma != 0 and start != 0 and chongfu_start == False:

            msg = gaussian(record, sigma, size)
            old = msg
            if ii % 4 == 0:
                print(msg)

        if jilu_start and not chongfu_start:
            jilu.append(msg)
            jilu_index += 1
            print('record', jilu[-1])

        if chongfu_start:

            shunxu.extend(list(np.arange(0, -jilu_index + 1, -1)))
            shunxu.extend(list(np.arange(-jilu_index + 1, 0, 1)))

            msg2 = jilu[shunxu[chongfu_index]]
            print(msg2)
            time.sleep(0.1)
            chongfu_index += 1

        if wuming < 100 and ii > 200 and zhongzhi > 100:
            plot_3D(record[0], record[1], record[2], record[3], ii)
            print('结束')
            break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img, f'fps:{int(fps)}', [15, 25],
    #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)