import cv2 as cv
import numpy as np
import HandtrackingModule as Handtrack
import time
import subprocess
import PySimpleGUI as sg
import math

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

#====================================================================================

def my_main():
    ExecuteThisAdb("adb disconnect")
    ExecuteThisAdb("adb connect 192.168.43.6:5555")

def UP():
    ExecuteThisAdb("adb shell input keyevent 19")

def DOWN():
    ExecuteThisAdb("adb shell input keyevent 20")

def LEFT():
    ExecuteThisAdb("adb shell input keyevent 21")

def RIGHT():
    ExecuteThisAdb("adb shell input keyevent 22")

def ExecuteThisAdb(Code):
    cmd = subprocess.Popen(Code, stdout = subprocess.PIPE, stderr = None, shell=True)
    print(cmd.communicate()[0]) 

#====================================================================================

width, height = 1920,1080

Capture = cv.VideoCapture(1)
Capture.set(3,width)
Capture.set(4,height)

pTime = 0

detector = Handtrack.handDetector(detectionCon=0.1)

left_rect_coord = []
right_rect_coord = []
up_rect_coord = []
down_rect_coord = []


for i in range(0, 200):
    for j in range(400,880):
        left_rect_coord.append((i,j))

for i in range(1520,1920):
    for j in range(200,880):
        right_rect_coord.append((i,j))

for i in range(410,1510):
    for j in range(0,250):
        up_rect_coord.append((i,j))

for i in range(410, 1510):
    for j in range(830,1080):
        down_rect_coord.append((i,j))

my_main()

while True:
    
    success, frame = Capture.read()
    
    frame = detector.findHands(frame)
    #Draw Rectangles
    
    #left triangle
    cv.rectangle(frame, (0, 200), (400,880), (0,255,0), 1)
    
    #right triangle
    cv.rectangle(frame, (1520,200), (1920,880), (0,255,0), 1)
    
    #up triangle
    cv.rectangle(frame, (410,0), (1510,250), (0,255,0), 1)

    #down traingle
    cv.rectangle(frame, (410, 830), (1510,1080), (0,255,0), 1)


    list_of_positions = detector.findPosition(frame, draw=False)
    
    if len(list_of_positions) > 20:
        thumb = [list_of_positions[4][1], list_of_positions[4][2]]
        first_finger = (list_of_positions[8][1], list_of_positions[8][2])
        second_finger = (list_of_positions[12][1], list_of_positions[12][2])
        third_finger = (list_of_positions[16][1], list_of_positions[16][2])
        fourth_finger = (list_of_positions[20][1], list_of_positions[20][2])

        cv.circle(frame, (thumb[0],thumb[1]), 15, (255,0,255), cv.FILLED)
        cv.circle(frame, (first_finger[0],first_finger[1]), 15, (255,0,255), cv.FILLED)
        cv.circle(frame, (second_finger[0],second_finger[1]), 15, (255,0,255), cv.FILLED)
        cv.circle(frame, (third_finger[0],third_finger[1]), 15, (255,0,255), cv.FILLED)
        cv.circle(frame, (fourth_finger[0],fourth_finger[1]), 15, (255,0,255), cv.FILLED)

        
        if  first_finger in left_rect_coord:
            LEFT()

        if  second_finger in right_rect_coord:
            RIGHT()

        if  third_finger in up_rect_coord:
            UP()

        if  fourth_finger in down_rect_coord:
            DOWN()


    cTime = time.time()
    fps = 1/(pTime-cTime)
    pTime = cTime
    cv.putText(frame, f'FPS: {int(fps)}', (40,50), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    frame75 = rescale_frame(frame)
    cv.imshow("hand detection", frame75) 
    if cv.waitKey(20) & 0Xff==ord(' '):
        break


   




