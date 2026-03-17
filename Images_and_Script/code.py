# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:41:52 2025

@author: Laptop
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_signs(img, x):
    confirmed_signs = []
    
    nice = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    saved_copy = nice.copy()
    
    plt.title("Image "+str(x), fontsize = 12, loc='center', pad = 20)
    plt.axis('off') 
    
    plt.subplot(2,3,1)
    plt.imshow(nice)  
    plt.axis('off') 
    plt.title("Original Image", fontsize = 8, loc='center', pad = 0)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.medianBlur(img, 9)
        
    lower_bound1 = np.array([0, 100, 100])
    upper_bound1 = np.array([10, 255, 255])
    red1 = cv2.inRange(img, lower_bound1, upper_bound1)
    
    lower_bound2 = np.array([160, 100, 100])
    upper_bound2 = np.array([179, 255, 255])
    red2 = cv2.inRange(img, lower_bound2, upper_bound2)
    
    red = cv2.bitwise_or(red1, red2)
    
    final = cv2.bitwise_and(img, img, mask=red)
        
    h, s, v = cv2.split(final)
    plt.subplot(2,3,2)
    plt.imshow(s, cmap = 'gray')  
    plt.axis('off') 
    plt.title("Red Elements", fontsize = 8, loc='center', pad = 0)

    _, s =  cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
    
    plt.subplot(2,3,3)
    plt.imshow(s, cmap = 'gray')  
    plt.axis('off') 
    plt.title("Red Filtered", fontsize = 8, loc='center', pad = 0)
    
    s = cv2.medianBlur(s, 5)
    
    circles = cv2.HoughCircles(
        s,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=200,      
        param1=100,         
        param2=40,       
        minRadius=10,       
        maxRadius=min(s.shape[0], s.shape[1])//2
    )
    
    if circles is not None:
        circles = np.int32(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(nice, (i[0], i[1]), i[2], (0, 255, 0), 10)
            cv2.circle(nice, (i[0], i[1]), 2, (0, 0, 255), 3)

            red = s.copy()
            h1, s1, v1 = cv2.split(img)
            x = i[0]
            y = i[1]
            r = i[2]
                    
            red_pixels = 0
            other_colors = 0
            total = 0
            
            for a in range(x-r, x+r):
                for b in range(y - r, y + r):
                    if min(x-r, y-r) >= 0 and x+r < img.shape[1] and y+r < img.shape[0]:
                        if (a-x)**2 + (b-y)**2 <= r**2:
                            total += 1
                            if red[b][a] == 255:
                                red_pixels += 1
                            elif s1[b][a] > 50:
                                other_colors += 1
            """
            if red_pixels > 0.4 * total:
                print("too much red")

            elif other_colors > 0.5 * total:
                print("weird colors in the circle", other_colors/total)
            else:
                #print(other_colors/total)
                confirmed_signs.append((x, y, r))
            
            if red_pixels < 0.4 * total and other_colors < 0.55 * total:
                confirmed_signs.append((x, y, r))
            """
                
    plt.subplot(2,3,4)
    plt.imshow(nice)  
    plt.axis('off') 
    plt.title("Detected Circles", fontsize = 8, loc='center', pad = 0)

    canvas = saved_copy.copy()
    for each in confirmed_signs:
        x = each[0]
        y = each[1]
        r = each[2]
        cv2.circle(canvas, (x, y), r, (0, 255, 0), 10)

    plt.subplot(2,3,5)
    plt.imshow(canvas)  
    plt.axis('off') 
    plt.title("Detected Signs", fontsize = 8, loc='center', pad = 0)
        
    plt.show()
    
fig = plt.figure(figsize=(8, 6))

"""
for x in range(1, 26):
    path = "img" + str(x)+".jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    nice = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(5, 5, x)
    plt.imshow(nice)  
    plt.axis('off') 
    plt.title("Image "+str(x), fontsize = 8, loc='center', pad = 0)
    
    plt.imshow(nice)
    plt.axis('off') 
    plt.title("Image "+str(x), fontsize = 8, loc='center', pad = 0)
plt.show()
"""

for x in range(1, 26):
    path = "img" + str(x)+".jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    detect_signs(img, x)
