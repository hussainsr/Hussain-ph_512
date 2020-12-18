# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:27:27 2020

@author: hussa
"""
import imageio
#from PIL import Image, ImageDraw

im = []
for i in range(20,80):
    im.append(imageio.imread('part4'+str(i)+'.png'))
    
imageio.mimsave('part4(2).gif',im, duration = 0.1)    