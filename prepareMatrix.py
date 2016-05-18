# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:59:04 2016

@author: Tomasz Sosnowski
"""
# spawny: -780;400 i -250;400

import numpy as np

def returnColor(img,y,x,n,m):
    imgR=img[0]
    imgG=img[1]
    imgB=img[2]
    y1=(imgR.shape[0]/n)*y
    y2=min(((imgR.shape[0]/n)*(y+1)),imgR.shape[0])
    x1=(imgR.shape[1]/m)*x
    x2=min(((imgR.shape[1]/m)*(x+1)),imgR.shape[1])
    interestingR=imgR[y1:y2,x1:x2]
    interestingG=imgG[y1:y2,x1:x2]
    interestingB=imgB[y1:y2,x1:x2]
    rmean=np.mean(interestingR)
    gmean=np.mean(interestingG)
    bmean=np.mean(interestingB)
    if (rmean<50 and gmean<50 and bmean<50):
        return 1
    if (rmean>200 and gmean>200 and bmean>200):
        return (-1)
    return 0

def prepareMatrix(img,n,m):
    M = np.zeros((n,m))
    for i in range(0,n):
        for j in range(0,m):
            M[i,j]=returnColor(img,i,j,n,m)
    return M