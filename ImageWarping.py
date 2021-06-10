import cv2
import numpy as np
# import os
imgWidth, imgHeight = 480, 720

def getContours(img):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    cnt = contours[0]
    #imgContour - the img to which these contours should be drawn
    #cnt - the contour values itself,
    #-1 specifies all the contours should be drawn
    #3 - thickness of contours
    cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
    #True specifies a closed figure
    perimeter = cv2.arcLength(cnt, True)
    #approx gives us all the corners of the figure or contour
    approx = cv2.approxPolyDP(cnt, 0.04*perimeter, True)
    no_of_corners = len(approx)
    return approx


def setOrder(points):
    points = np.squeeze(points)
    new_points = np.zeros_like(points)
    points_sum = np.sum(points, axis=1)
    points_diff = np.diff(points, axis =1 )
    new_points[0] = points[np.argmin(points_sum)]
    new_points[3] = points[np.argmax(points_sum)]
    new_points[1] = points[np.argmin(points_diff)]
    new_points[2] = points[np.argmax(points_diff)]
    new_points = new_points.reshape(4,1,2)
    return new_points


def getWarped(img, biggest):

    biggest = setOrder(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    finalImage = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
    return finalImage


def imagePreprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 200, 200)
    kernel = np.ones((5,5))
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    return img


img = cv2.imread('Resources/number.jpeg')
img = cv2.resize(img, (imgWidth, imgHeight))
imgContour = img.copy()
processedImage = imagePreprocess(img)
biggest = getContours(processedImage)
warpedImage = getWarped(img, biggest)

#cv2.imshow('Image',processedImage)
cv2.imshow('Image', warpedImage)
cv2.waitKey(0)




