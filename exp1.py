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
import pyaudio  # 使用pyaudio库可以进行录音，播放，生成wav文件
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from socket import *

class angle():
    def __init__(self, t1):
        self.t1 = t1

    def sin(self):
        rad = self.t1 / 180 * np.pi
        s = np.sin(rad)
        return s

    def cos(self):
        rad = self.t1 /180 * np.pi
        c = np.cos(rad)
        return c

def Ti(alpha, a, theta, d):
    R1 = np.mat([[1, 0, 0, 0],
                 [0, angle(alpha).cos(), -angle(alpha).sin(), 0],
                 [0, angle(alpha).sin(), angle(alpha).cos(), 0],
                 [0, 0, 0, 1]])

    T1 = np.mat([[1, 0, 0, a],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

    R2 = np.mat([[angle(theta).cos(), -angle(theta).sin(), 0, 0],
                 [angle(theta).sin(), angle(theta).cos(), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

    T2 = np.mat([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, d],
                 [0, 0, 0, 1]])

    return R1 * T1 * R2 * T2


def plot_3D(qlist, wlist, elist, rlist, l):
    plot_x = []
    plot_y = []
    plot_z = []
    for i in range(50, l):
        q = qlist[i]
        w = wlist[i]
        e = elist[i]
        r = rlist[i]

        d1, d2, l3, l4, d5 = 10, 10, 10, 10, 10
        theta1, theta2, theta3, theta4 = 0 - q + 90, 0 - w + 90, 0 - e + 90, 0 - r + 90
        T1 = Ti(0, 0, theta1, d1)
        T2 = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]]) * np.mat([[1, 0, 0, 0],
                    [0, angle(90).cos(), -angle(90).sin(), 0],
                    [0, angle(90).sin(), angle(90).cos(), 0],
                    [0, 0, 0, 1]]) * np.mat([[angle(theta2).cos(), -angle(theta2).sin(), 0, 0],
                                            [angle(theta2).sin(), angle(theta2).cos(), 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])

        T3 = Ti(0, l3, theta3, 0)
        T4 = Ti(0, l4, theta4, 0)
        T5 = np.mat([[1, 0, 0, d5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) * np.mat(
            [[angle(-90).cos(), angle(-90).sin(), 0, 0],
            [angle(-90).sin(), angle(-90).cos(), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]) * np.mat([[1, 0, 0, 0],
                                    [0, angle(-90).cos(), -angle(-90).sin(), 0],
                                    [0, angle(-90).sin(), angle(-90).cos(), 0],
                                    [0, 0, 0, 1]])

        i5 = np.mat([0, 0, 10, 1]).T
        i0 = T1 * T2 * T3 * T4 * T5 * i5
        plot_x.append(float(i0[0]))
        plot_y.append(float(i0[1]))
        plot_z.append(float(i0[2]))

        # return plot_x, plot_y, plot_z

    fig = plt.figure(figsize=(100, 70))
    ax = plt.axes(projection="3d")

    # Creating a plot using the random datasets
    ax.scatter3D(plot_x, plot_y, plot_z, color="red")
    ax.plot(plot_x, plot_y, plot_z, color="blue")
    plt.show()

if __name__ == '__main__':
    chongfu_start = False
    jilu_start = False

    udp_socket = socket(AF_INET, SOCK_DGRAM)

    dest_addr = ('192.168.137.39', 9924)
    hand()
    thread1 = threading.Thread(name='t1', target=hand)
    thread2 = threading.Thread(name='t2', target=speak)

    thread1.start()   #启动线程1
    thread2.start()   #启动线程2

    while True:
        print('huitu')
        if huitu:
            plot_3D(record[0], record[1], record[2], record[3], ii)
            break
        time.sleep(0.1)