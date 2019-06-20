# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageColor
from PIL import ImageEnhance
import imageio
import cv2

# PART 1

img = imageio.imread('rihanna.jpg')
#rihanna.jpg
size1 = img.shape[0]
size2 = img.shape[1]

half1 = int(size1/2)
half2 = int(size2/2)

#sag ust parca
image1 = Image.fromarray(img[0:half1, half2:size2])

br = ImageEnhance.Brightness(image1)
br = br.enhance(1.3)

image1_arr = np.asarray(br)

img[0:half1, half2:size2] = image1_arr

#sol alt parca
image2 = Image.fromarray(img[half1:size1, 0:half2])

cnt = ImageEnhance.Contrast(image2)
cnt = cnt.enhance(1.6)

image2_arr = np.asarray(cnt)

img[half1:size1, 0:half2] = image2_arr

#sag alt parca
image3 = Image.fromarray(img[half1:size1, half2:size2])

st = ImageOps.equalize(image3)
image3_arr = np.asarray(st)

img[half1:size1, half2:size2] = image3_arr
plt.imshow(img) 



#HISTOGRAMS
color = ('b','g','r')
fig=plt.figure(figsize=(12, 10))
fig.add_subplot(4, 1, 1)
for channel,col in enumerate(color):
    histr = cv2.calcHist([img[0:half1, 0:half2]],[channel],None,[256],[0,256])
    #histr = cv2.calcHist([img[0:half1, 0:half2]],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale quadrant-1')
fig.add_subplot(4, 1, 2)
for channel,col in enumerate(color):
    histr = cv2.calcHist([image1_arr],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale quadrant-2')
fig.add_subplot(4, 1, 3)
for channel,col in enumerate(color):
    histr = cv2.calcHist([image2_arr],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale quadrant-3')
fig.add_subplot(4, 1, 4)
for channel,col in enumerate(color):
    histr = cv2.calcHist([image3_arr],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale quadrant-4')
plt.subplots_adjust(hspace=0.5)
plt.show()



## PART 2

path = "C:/Users/bgulk/Desktop/rihanna.jpg"
img = Image.open(path)
members = [(0,0)] * 25
width = np.shape(img)[0]
height = np.shape(img)[1]
newimg = Image.new("RGB",(width,height),"white")

for i in np.arange(2,width-2):
    for j in np.arange(2,height-2):
        a=0
        for k in np.arange(-2,3):
            for l in np.arange(-2,3):
                members[a] = img.getpixel((int(i+k),int(j+l)))
                a+=1         
        members.sort()
        newimg.putpixel((i,j),(members[13]))
plt.title('5x5 median filter')
fig.add_subplot(1, 1, 1)
plt.imshow(newimg)



## PART 3

img = imageio.imread('rihanna.jpg')
img = img[:,:,0]

fig=plt.figure(figsize=(10, 10))

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
fig.add_subplot(2, 2, 1)
plt.title('Magnitude Spectrum')
plt.imshow(magnitude_spectrum, 'gray')

rows = img.shape[0]
cols = img.shape[1]
crow,ccol = rows//2 , cols//2
fshift[crow-90:crow+90, ccol-90:ccol+90] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
fig.add_subplot(2, 2, 2)
plt.title('High Frequency Components')
plt.imshow(img_back, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Original image')
plt.imshow(img, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Image - high frequency')
plt.imshow(img-img_back, 'gray')