from django.shortcuts import render
import tkinter
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import sleep

# Create your views here.
def home(request):
    return render(request, 'home.html')

def showResult(request):
    root =Tk()
    
    images =[]

    root.configure(background="white")
    root.title = "MRI SEGMENTATION"
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))
    print(root.filename)
    lnk= root.filename
    img = cv2.imread(lnk)
    # plt.imshow(img)
    im = Image.fromarray(img)
    x=im.save("static/startImage.jpeg")
    
    images.append(x)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # plt.imshow(gray)
    # plt.imshow(thresh)
    im = Image.fromarray(thresh)
    x=im.save("static/thresh.jpeg")
    
    images.append(x)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 4)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    #Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    # plt.imshow(markers)
    image = cv2.imread(lnk,0)
    height, width = image.shape
    canny = cv2.Canny(image,50,175)
    plt.imshow( canny)
    im = Image.fromarray(canny)
    x=im.save("static/result.jpeg")

    images.append(x)


    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im = Image.fromarray(canny)
    x=im.save("static/result2.jpeg")
    
    images.append(x)


    cv2.drawContours(image, contours, -1 , (0,255,0),3)
    im = Image.fromarray(canny)
    x=im.save("static/result3.jpeg")
    
    images.append(x)
    
    root.mainloop()
    print(images)
    stuff={
        'images': images,
    }
    return render(request, 'display.html',stuff)
