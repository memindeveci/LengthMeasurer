import cv2
import numpy as np

def getContours(img, cThr=[100, 100], showCanny = False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel=np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThr = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny: cv2.imshow('Canny', imgThr)

    contours, hierarchy = cv2.findContours(imgThr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approximate = cv2.approxPolyDP(i, 0.02*peri, True)
            boundBox = cv2.boundingRect(approximate)
            if filter > 0:
                if len(approximate) == filter:
                    finalCountours.append(len(approximate), area, approximate, boundBox, i)
            else:
                finalCountours.append([len(approximate), area, approximate, boundBox, i])
    finalCountours = sorted(finalCountours, key = lambda x:x[1], reverse=True)

    if draw:
        for contours in finalCountours:
            cv2.drawContours(img, contours[4], -1, (0, 255, 255), 3)

    return img, finalCountours

def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    differentiation = np.diff(myPoints, axis = 1)
    myPoints[1] = myPoints[np.argmin(differentiation)]
    myPoints[2] = myPoints[np.argmin(differentiation)]
    return myPointsNew


def warpImg (img, points, width, height, pad=20):
    #print(points)
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))
    imgWarp = imgWarp[pad: imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]

    return imgWarp

def findDistance(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5


