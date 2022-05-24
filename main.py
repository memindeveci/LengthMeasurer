import cv2
import numpy as np

import Utils

cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    success, img = cap.read()

    imgContours, finalCountours = Utils.getContours(img, minArea=50000, filter=4)

    if len(finalCountours) != 0:
        biggest = finalCountours[0][2]
        #print(biggest)
        imageWarp = Utils.warpImg(img, biggest, 1000, 1000)
        cv2.imshow('WarpedImg', imageWarp)

        imgContours2, finalCountours2 = Utils.getContours(img, showCanny=True, minArea=5000, filter=4, draw=True, cThr=[50,50])


        if len(finalCountours) != 0:
            for obj in finalCountours:
                cv2.polylines(imgContours2, [obj[2]], True, (0,255,0), 2)
                nPoints = Utils.reorder(obj[2])
                w= round((Utils.findDistance(nPoints[0][0], nPoints[1][0])//10),1)
                h = round((Utils.findDistance(nPoints[0][0], nPoints[2][0]) // 10), 1)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]), (255,0,255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]), (255,0,255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(w), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255,0,255))
                cv2.putText(imgContours2, '{}cm'.format(h), (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255,0,255))


        cv2.imshow('WarpedImg', imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Original', img)
    cv2.waitKey(1)
