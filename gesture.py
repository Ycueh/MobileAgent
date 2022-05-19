#bgr8转jpeg格式

import enum

import cv2

 

def bgr8_to_jpeg(value, quality=75):

    return bytes(cv2.imencode('.jpg', value)[1])

import cv2

import time

import demjson

import pygame
from aip import AipBodyAnalysis
from aip import AipSpeech
from PIL import Image, ImageDraw, ImageFont
import numpy
import ipywidgets.widgets as widgets

 

# 具体手势请看官方提供 https://ai.baidu.com/ai-doc/BODY/4k3cpywrv

hand={'One':'Number One','Five':'Number Five','Fist':'Fist','Ok':'OK',

      'Prayer':'Prayer','Congratulation':'Congratulation','Honour':'Honour',

      'Heart_single':'Heart_single','Thumb_up':'Thumb_up','Thumb_down':'Diss',

      'ILY':'I Love you','Palm_up':'Palm_up','Two':'Number 2',

      'Three':'Number Three','Four':'Number 4','Six':'Number 6','Seven':'Number 7',

      'Eight':'Number 8','Nine':'Number 9','Rock':'Rock','Face':'Face'}

 

# 下面的key要换成自己的

""" 人体分析 APPID AK SK """
APP_ID = '26065203'
API_KEY = 'cl0IBeIVLOc3HGy7GpOSzHng'
SECRET_KEY = 'anSxZGhTQRdRIiCseSatMAlfPr2Gnkmw'
#camera = PiCamera()
client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
g_camera = cv2.VideoCapture(0)
g_camera.set(3, 640)
g_camera.set(4, 480)
g_camera.set(5, 30)  #设置帧率
g_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
g_camera.set(cv2.CAP_PROP_BRIGHTNESS, 60) #设置亮度 -64 - 64  0.0
g_camera.set(cv2.CAP_PROP_CONTRAST, 50) #设置对比度 -64 - 64  2.0
g_camera.set(cv2.CAP_PROP_EXPOSURE, 156) #设置曝光值 1.0 - 5000  156.0

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "simhei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

while True:
    ret, frame = g_camera.read()
    if ret:
        raw = str(client.gesture(bgr8_to_jpeg(frame)))
        text = demjson.decode(raw)
        try:
            res = text['result'][0]['classname']
        except:
            cv2.putText(frame, 'No Objects', (250,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2, cv2.LINE_AA)
        else:
            print('Result：' + hand[res])
            cv2.putText(frame, hand[res], (250,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("result", frame)
