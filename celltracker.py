#!/usr/bin/python
'''
E. coli bacteria tracking suite
----------------------------------
Author: Guillaume Lambert

'''

import numpy as np
import re
import cv2 as cv
import matplotlib.path as pa
import matplotlib.pylab as plt
import os as os
import time as time
import sys as sys
import hungarian as hun

#Progress bar, from http://stackoverflow.com/a/6169274


def startProgress(title):
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    globals()["progress_x"] = 0


def progress(x):
    x = np.floor(x*40.0)
    sys.stdout.write("#"*(int(x) - globals()["progress_x"]))
    sys.stdout.flush()
    globals()["progress_x"] = x


def endProgress():
    sys.stdout.write("#"*(40 - globals()["progress_x"]))
    sys.stdout.write("]\n")
    sys.stdout.flush()


def sort_nicely(l):
    """ Sort the given iterable in the way that humans expemt."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c)
                                for c in re.split('([0-9]+)',  key)]
    return sorted(l, key=alphanum_key)


def unique_rows(a):
    """ Find the unique rows in an array """
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def smooth(x, windowLength=3, type='boxcar'):
    """
    xs=smooth(x,windowLength,type='boxcar') is a simple moving
    average smoothing function
    Type can be:
        -'boxcar'
        -'gaussian'
        -'triang'
    """
    from scipy import signal
    x = np.double(x)
    s = np.r_[2*x[0]-x[windowLength:1:-1], x, 2*x[-1]-x[-1:-windowLength:-1]]

    if type == 'boxcar':
        w = np.ones(windowLength, 'd')
    elif type == 'gaussian':
        w = signal.gaussian(len(x), windowLength)
    elif type == 'triang':
        w = signal.triang(2*windowLength)
    y = np.convolve(w/w.sum(), s, mode='valid')
    #y=np.convolve(w/w.sum(),x,mode='same')
    return y[((windowLength-1)/2):-((windowLength-1)/2)]


def findAxes(m):
    """ Find the Minor and Major axis of an ellipse having moments m """
    Major = 2*(2*(((m['mu20'] + m['mu02']) +
                  ((m['mu20'] - m['mu02'])**2 +
                   4*m['mu11']**2)**0.5))/m['m00'])**0.5
    Minor = 2*(2*(((m['mu20'] + m['mu02']) -
              ((m['mu20'] - m['mu02'])**2 +
               4*m['mu11']**2)**0.5))/m['m00'])**0.5
    return Major, Minor


def anormalize(x):
    """ Normalize an array """
    y = x/np.sum(x)
    return y


def bpass(img, lnoise, lobject):
    '''
    imgOut = bpass(img,lnoise,lobject) return an
    image filtered according to the lnoise (size of the noise)
    and the lobject (typical size of the object in the image) parameter.

    The script has been translated from the bpass.m script developed
    by John C. Crocker and David G. Grier.'''

    from scipy import signal

    lnoise = np.double(lnoise)
    lobject = np.double(lobject)
    image_array = np.double(img)

    #Create the gaussian and boxcar kernels
    gauss_kernel = anormalize(signal.gaussian(np.floor(10*lnoise) + 1, lnoise))
    boxcar_kernel = anormalize(signal.boxcar(2*lobject))
    #Apply the filter to the input image
    gconv0 = cv.filter2D(np.transpose(image_array), -1, gauss_kernel)
    gconv = cv.filter2D(np.transpose(gconv0), -1, gauss_kernel)
    bconv0 = cv.filter2D(np.transpose(image_array), -1, boxcar_kernel)
    bconv = cv.filter2D(np.transpose(bconv0), -1, boxcar_kernel)
    #Create the filtered image
    filtered = gconv - bconv
    #Remove the values lower than zero
    filtered[filtered < 0] = 0
    #Output the final image
    imgOut = filtered.copy()
    return imgOut


def mat2gray(img, scale):
    '''
    imgOut = mat2gray(img, scale) return a rescaled matrix from 1 to scale
    '''

    imgM = img - img.min() + 1

    imgOut = imgM*np.double(scale)/imgM.max()

    return imgOut


def unsharp(img, sigma=5, amount=10):
    '''
    imgOut = unshapr(img, sigma=5, amount=10) create an unsharp
    operation on an image.
    If amount=0, create a simple gaussian blur with size sigma.
    '''
    if sigma:
        img = np.double(img)
        #imgB=cv.GaussianBlur(img, (img.shape[0]-1, img.shape[1]-1), sigma)
        imgB = cv.GaussianBlur(img, (img.shape[0]-(img.shape[0]+1) % 2,
                               img.shape[1]-(img.shape[1]+1) % 2), sigma)
        imgS = img*(1 + amount) + imgB*(-amount)
        if amount:
            return imgS
        else:
            return imgB
    else:
        return img


def regionprops(bwImage, scaleFact=1):
    '''Replicate MATLAB's regionprops script.
    STATS = regionprops(bwImage, scaleFact=1) returns the following
    properties in the STATS array:
    STATS=[xPosition, yPosition, MajorAxis,
           MinorAxis, Orientation, Area, Solidity]
    '''
    #import the relevant modules
    #Find the contours
    bwI = np.uint8(bwImage.copy())
    csr, _ = cv.findContours(bwI.copy(),
                             mode=cv.RETR_TREE,
                             method=cv.CHAIN_APPROX_SIMPLE)
    numC = int(len(csr))
    #Initialize the variables
    k = 0
    for i in range(numC):
        if len(csr[i]) >= 5:
            k = k + 1
    majorAxis = np.zeros((k, 1), dtype=float)
    minorAxis = np.zeros((k, 1), dtype=float)
    Orientation = np.zeros((k, 1), dtype=float)
    EllipseCentreX = np.zeros((k, 1), dtype=float)
    EllipseCentreY = np.zeros((k, 1), dtype=float)
    Area = np.zeros((k, 1), dtype=float)
    Solidity = np.zeros((k, 1), dtype=float)
    k = 0
    for i in range(numC):
        if len(csr[i]) >= 5:
            #Orientation
            centroid, axes, angle = cv.fitEllipse(csr[i])
            Orientation[k] = angle
            #Major,Minor axes
            #Centroid
            m = cv.moments(csr[i])
            if m['m00'] > 0:
                centroid = (m['m10']/m['m00'],
                            m['m01']/m['m00'])
                majorAxis[k], minorAxis[k] = findAxes(m)
                EllipseCentreX[k] = centroid[0]
                EllipseCentreY[k] = centroid[1]
                #Solidity&Area
                Area[k] = m['m00']
                # CONVEX HULL stuff
                # convex hull vertices
                ConvexHull = cv.convexHull(csr[i])
                ConvexArea = cv.contourArea(ConvexHull)
                # Solidity : =  Area/ConvexArea
                Solidity[k] = np.divide(Area[k], ConvexArea)
            k = k + 1
    allData = np.zeros((k, 7))
    allData[:, 0] = EllipseCentreX.squeeze()/scaleFact
    allData[:, 1] = EllipseCentreY.squeeze()/scaleFact
    allData[:, 2] = majorAxis.squeeze()/scaleFact
    allData[:, 3] = minorAxis.squeeze()/scaleFact
    allData[:, 4] = Orientation.squeeze()
    allData[:, 5] = Area.squeeze()/scaleFact**2
    allData[:, 6] = Solidity.squeeze()
    return np.double(allData)


def bwlabel(bwImg):
    '''Replicate MATLAB's bwlabel function.
    labelledImg = bwlabel(bwImg) takes a black&white image and
    returns an image where each connected region is labelled
    by a unique number.
    '''

    #Import relevant modules
    #Change the type of image
    bwImg2 = np.uint8(bwImg.copy())
    bw = np.zeros(bwImg2.shape)
    #Find the contours,  count home many there are
    csl, _ = cv.findContours(bwImg2.copy(),
                             mode=cv.RETR_TREE,
                             method=cv.CHAIN_APPROX_SIMPLE)
    numC = int(len(csl))
    #Label each cell in the figure
    k = 0
    for i in range(numC):
        if len(csl[i]) >= 5:
            k = k + 1
            cv.drawContours(bw, csl, i, k, thickness=-1)
        else:
            cv.drawContours(bw, csl, i, 0, thickness=-1)
    return np.uint16(bw)


def traceOutlines(trT, dist, size):
    '''
    ImgOut = traceOutlines(trT) Trace the outline of every cell
    in the array trT (trT should contain the tr data of a single time).
    '''

    bw = np.zeros(size)
    csl = []
    for pt in trT:
        boundPt = getBoundingBox(pt, dist)
        dp = np.array(([[[int(round(boundPt[0, 0])),
                          int(round(boundPt[0, 1]))]]]), dtype=np.int32)
        for p in boundPt:
            dp = np.vstack((dp, np.array(([[[int(round(p[0])),
                                             int(round(p[1]))]]]),
                                         dtype=np.int32)))
        csl.append(dp)
    numC = int(len(csl))
    for i in range(numC):
        if len(csl[i]) > 0:
            cv.drawContours(bw, csl, i, int(trT[i][3]), thickness=-1)
        else:
            cv.drawContours(bw, csl, i, 0, thichness=-1)
    return bw


def removeSmallBlobs(bwImg, bSize=10):
    '''
    imgOut = removeSmallBlobs(bwImg, bSize) removes processes a binary image
    and removes the blobs which are smaller than bSize (area).
    '''
    bwImg2 = np.uint8(bwImg.copy())
    bw = np.zeros(bwImg2.shape)

        #Find the contours,  count home many there are
    csl, _ = cv.findContours(bwImg2.copy(),
                             mode=cv.RETR_TREE,
                             method=cv.CHAIN_APPROX_SIMPLE)
    numC = int(len(csl))
    #Label each cell in the figure which are smaller than bSize
    for i in range(numC):
        area = cv.contourArea(csl[i])
        if area <= bSize:
            cv.drawContours(bw,  csl,  i,  1,  thickness=-1)
        else:
            cv.drawContours(bw,  csl,  i,  0,  thickness=-1)
    maskBW = 1-bw
    return np.uint8(bwImg2*maskBW)


def floodFill(imgIn, seedPt, pixelValue):
    '''
    This script perform a flood fill,  starting from seedPt.
    '''
    labelledImg = bwlabel(imgIn)

    regionID = labelledImg[seedPt[0], seedPt[1]]

    bwImg0 = imgIn.copy()
    bwImg0[labelledImg == regionID] = pixelValue

    return bwImg0


def avgCellInt(rawImg, bwImg):
    '''
    STATS = avgCellInt(rawImg, bwImg) return an array containing
    the pixel value in rawImg of each simply connected region in bwImg.
    '''
    bwImg0 = np.uint8(bwImg.copy())
    bw = np.zeros(bwImg0.shape)

    csa, _ = cv.findContours(bwImg0,
                             mode=cv.RETR_TREE,
                             method=cv.CHAIN_APPROX_SIMPLE)
    numC = int(len(csa))
    k = 0
    for i in range(0, numC):
        if len(csa[i]) >= 5:
            k = k + 1
    avgCellI = np.zeros((k + 1, 1), dtype=float)
    k = 0
    for i in range(0, numC):
        if len(csa[i]) >= 5:
            # Average Pixel value
            bw = np.zeros(bwImg0.shape)
            k = k + 1
            cv.drawContours(bw, csa, i, 1, thickness=-1)
            regionMask = (bw == (1))
            avgCellI[k] = np.sum(rawImg*regionMask)/np.sum(regionMask)
    return np.double(avgCellI)


def segmentCells(bwImg, iterN=1):
    '''
    imgOut = segmentCells(bwImg, propIndex, pThreshold, iterN = 2) applies
    a watershed transformation to bwImg.
    '''
    labelImg = bwlabel(bwImg).copy()
    lowImg = np.uint16(labelImg > 0)
    markers = cv.distanceTransform(np.uint8(lowImg), cv.cv.CV_DIST_L2, 5)
    markers32 = np.int32(markers)

    rgbImg = np.zeros((np.size(lowImg, 0), np.size(lowImg, 1), 3))
    rgbImg[:, :, 0] = np.uint8(lowImg > 0)
    rgbImg[:, :, 1] = np.uint8(lowImg > 0)
    rgbImg[:, :, 2] = np.uint8(lowImg > 0)

    cv.watershed(np.uint8(rgbImg), markers32)
    m = cv.convertScaleAbs(markers32)-1
    m = dilateConnected(m, iterN)
    maskImg = (np.ones(((np.size(lowImg, 0),
               np.size(lowImg, 1)))) - m + lowImg) == 1
    segImg = np.multiply(maskImg, labelImg) > 0

    return np.double(segImg)


def fragmentCells(bwImg, propIndex, thres, iterN=1):
    '''

    '''

    labelImg = bwlabel(bwImg).copy()
    lowImg = np.double(np.zeros((np.size(labelImg, 0), np.size(labelImg, 1))))
    lowSIndex = np.nonzero(propIndex < thres)

    if lowSIndex[0].any():
        for id in np.transpose(lowSIndex):
            lowImg = lowImg + np.double(labelImg == (id + 1))
        lowImg = cv.erode(np.uint8(lowImg), None, iterations=iterN)
        m = segmentCells(lowImg)

        m = dilateConnected(m, iterN)

        for id in np.transpose(lowSIndex):
            bwImg = bwImg - np.double(labelImg == (id + 1))
        return np.uint8(bwImg + m)

    else:
        return bwImg


def dilateConnected(imgIn, nIter):
    """
    imgOut = dilateConnected(imgIn, nIter) dilates a binary image
    while preserving the number of simply connected domains.

    nIter is the dilation factor
    (number of times the dilate function is applied)
    """
    bwImgD = np.uint8(imgIn.copy())
    imgOut = np.double(imgIn*0)
    bwLD = bwlabel(bwImgD)
    for i in range(1, bwLD.max() + 1):
        imgOut = imgOut + np.double(cv.dilate(np.uint16(bwLD == i),
                                    None, iterations=(nIter + 2)))

    dilImg = cv.dilate(bwImgD, None, iterations=nIter)
    skelBnd = skeletonTransform(np.uint8(imgOut > 1))
    skelBnd = cv.dilate(skelBnd, None, iterations=1)

    imgOut = np.double(dilImg) - skelBnd*(bwLD == 0)
    imgOut = imgOut > 0
    return np.double(imgOut)


def skeletonTransform(bwImg):
    '''
    Generate the skeleton transform of a binary image
    Based on:
    http://opencvpython.blogspot.com/2012/05/
               skeletonization-using-opencv-python.html
    '''
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    img = np.uint8(bwImg.copy())
    skel = img*0
    while not done:
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = img - temp
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        if not cv.countNonZero(img):
            done = True
    #Remove isolated pixels

    skel = skel - cv.filter2D(skel, -1, np.array([[-9, -9, -9],
                                                  [-9, 1, -9],
                                                  [-9, -9, -9]], dtype=float))

    return skel


def labelOverlap(img1, img2, fraction=False):
    '''
    areaList = labelOverlap(img1, img2) finds the overlapping
    cells between two images.

    Returns areaList as [label1 label2 area]
    img1 and img2 should be generated with bwlabel.
    '''
    #Find the overlapping indices
    overlapImg = np.uint8((img1 > 0)*(img2 > 0))
    index1 = overlapImg*np.uint16(img1)
    index2 = overlapImg*np.uint16(img2)

    #Store the index couples in a list
    indexList = np.vstack([index1.flatten(0), index2.flatten(0)])

    #remove the [0, 0] rows,  convert to 1D structured array
    indexList = np.uint16(np.transpose(indexList[:, indexList[0, :] > 0]))
    indexList = indexList.view(', '.join(2 * ['i2']))
    unique_vals,  indices = np.unique(indexList, return_inverse=True)
    if len(indices) > 0:
        counts = np.bincount(indices)
        numId = len(counts)
        areaList = np.zeros((numId, 3))
        for id in range(numId):
            areaList[id, 0] = unique_vals[id][0]
            areaList[id, 1] = unique_vals[id][1]
            if fraction:
                ar = counts[id]/np.double((index2 == unique_vals[id][1]).sum())
                areaList[id, 2] = ar.copy()
            else:
                areaList[id, 2] = counts[id]
    else:
        areaList = np.zeros((1, 3))
    return areaList


def matchIndices(dataList, matchType='Area'):
    '''
    linkList = matchIndices(dataList, matchType = 'Area') uses the
    Munkres algorithm to performs an assignment problem from the
    data in dataList = [id1, id2, distanceValue].

    *if matchType = 'Area',  it can be used to match the cell areas
    between successive frames (favorizes largest area value)

    *if matchType = 'Distance',  it can be used to match the cell
    positions between successive frames (favorises smallest distance value)
    '''
    linkList = [0, 0, 0]
    #Loop over each indices to find its related matches.
    for cellID in np.transpose(np.unique(dataList[:, 0])):
        stop = 0
        #check is dataList is empty
        if np.size(dataList, 0) > 0:
            matchData = dataList[dataList[:, 0] == cellID, :]
        else:
            matchData = []
        if np.size(matchData, 0) > 0:
            #Collect related cells
            while stop == 0:
                matchData0 = np.array(matchData)
                for i in np.transpose(np.unique(matchData[:, 1])):
                    matchData = np.vstack([matchData,
                                           dataList[dataList[:, 1] == i, :]])
                for i in np.transpose(np.unique(matchData[:, 0])):
                    matchData = np.vstack([matchData,
                                           dataList[dataList[:, 0] == i, :]])
                matchData = unique_rows(matchData)
                if np.array_equal(matchData, matchData0):
                    stop = 1
            #Create a label for each cell
            # (instead of the cell ID,  which may have gaps)
            k0 = 0
            assign0 = np.zeros((np.max(matchData[:, 0]) + 1))
            assign1 = np.zeros((np.max(matchData[:, 1]) + 1))
            for i in np.transpose(np.unique(matchData[:, 0])):
                k0 = k0 + 1
                assign0[i] = k0
            k1 = 0
            for i in np.transpose(np.unique(matchData[:, 1])):
                k1 = k1 + 1
                assign1[i] = k1

            k = np.max([k0, k1])
            #Create the linear assignment matrix
            A = np.zeros((k, k))
            for i in range(len(matchData[:, 0])):
                if matchType is 'Area':
                    indx1 = assign0[matchData[i, 0]]-1
                    indx2 = assign1[matchData[i, 1]]-1
                    A[indx1][indx2] = matchData[i, 2]
                elif matchType is 'Distance':
                    if matchData[i, 2] > 0:
                        indx1 = assign0[matchData[i, 0]]-1
                        indx2 = assign1[matchData[i, 1]]-1
                        A[indx1][indx2] = 1./matchData[i, 2]
            #Find matching elements with munkres algorithm
            #m = mk.Munkres()
            #inds = m.compute(-A)
            inds, inds0 = hun.lap(-A)
            inds2 = np.zeros((len(inds), 2))
            for kk in range(len(inds)):
                inds2[kk, 0] = kk
                inds2[kk, 1] = inds[kk]
            inds = inds2.copy()

            linkList0 = np.zeros((len(matchData[:, 0]), 3))
            #Put result into the linkList0 array
            for i in range(len(inds)):
                if ((inds[i][0] + 1) <= k0) & ((inds[i][1] + 1) <= k1):
                    linkList0[i, 0] = np.nonzero(assign0 ==
                                                 (inds[i][0] + 1))[0][0]
                    linkList0[i, 1] = np.nonzero(assign1 ==
                                                 (inds[i][1] + 1))[0][0]
                    if matchType is 'Area':
                        linkList0[i, 2] = A[inds[i][0]][inds[i][1]]
                    elif matchType is 'Distance':
                        if A[inds[i][0]][inds[i][1]] > 0:
                            linkList0[i, 2] = 1/A[inds[i][0]][inds[i][1]]

            linkList0 = linkList0[linkList0[:, 2] > 0, :]
            #Append data at the end of linkList
            linkList = np.vstack([linkList, linkList0])
            #Remove the matched data from the input list
            for id in np.transpose(linkList0[:, 0]):
                if np.size(dataList[:, 0] != id, 0) > 0:
                    dataList = dataList[dataList[:, 0] != id, :]
            for id in np.transpose(linkList0[:, 1]):
                if np.size(dataList[:, 1] != id, 0) > 0:
                    dataList = dataList[dataList[:, 1] != id, :]
    try:
        linkList = linkList[linkList[:, 0] > 0, :]
    except:
        linkList = np.zeros((1, 3))

    return linkList


def putLabelOnImg(fPath, tr, dataRange, dim, num):
    '''
    putLabelOnImg(fPath, tr, dataRange, dim) goes through the images in
    folder fPath over the time in dataRange and adds on top of each
    cell the values stored along dimension = dim of tr.
    '''
    #Initialize variables
    tifList = []
    trTime = splitIntoList(tr, 2)
    #Import and process image file
    fileList = sort_nicely(os.listdir(fPath))
    for file in fileList:
        if file.endswith('tif'):
            tifList.append(file)
    if not len(dataRange):
        dataRange = range(len(tifList))
    for t in dataRange:
        trT = trTime[t]
        if tifList:
            fname = tifList[t]
            print fPath + fname
            img = cv.imread(fPath + fname, -1)
            img = cv.transpose(img)
            bwImg = processImage(img, scaleFact=1, sBlur=0.5,
                                 sAmount=0, lnoise=1, lobject=7,
                                 boxSize=15, solidThres=0.65)
            plt.imshow(bwImg)
        else:
            if np.size(trT) > 1:
                img = traceOutlines(trT, 0.85, (950, 1200))
                plt.imshow(img)
        plt.hold(True)
        if np.size(trT) > 1:
            for cell in range(len(trT[:, 3])):
                '''plt.text(trT[cell, 0], trT[cell, 1],
                            str(trT[cell, dim])+',  '+str(trT[cell, 3]),
                        color = 'w', fontsize = 6)
                '''
                #boxPts = getBoundingBox(trT[cell, :])
                #plt.plot(boxPts[:, 0], boxPts[:, 1], 'gray', lw = 0.5)

                plt.text(trT[cell, 0], trT[cell, 1],
                         str(trT[cell, dim]), color='k', fontsize=6)
                for c in num:
                    if trT[cell, dim] == c:
                        plt.text(trT[cell, 0], trT[cell, 1],
                                 str(trT[cell, dim]), color='r', fontsize=6)

#       plt.clim((0, 30))
        plt.title(str(t))
        plt.hold(False)
        plt.xlim((0, 1250))
        plt.ylim((25, 220))

#       cv.imwrite(fPath+'Fig'+str(t)+'.jpg', np.uint8(bwI))
#       plt.savefig(fPath+"Fig"+str(t)+".png", dpi = (120))
        plt.show()
        plt.draw()
        time.sleep(0.5)
        plt.clf()


def processImage(imgIn, scaleFact=1, sBlur=0.5, sAmount=0, lnoise=1,
                 lobject=8, boxSize=15, solidThres=0.85):
    '''
    This is the core of the image analysis part of the tracking algorithm.
    bwImg = processImage(imgIn, scaleFact = 1, sBlur = 0.5, sAmount = 0,
                         lnoise=1, lobject = 8, boxSize = 15,
                         solidThres = 0.65)
    returns a black and white image processed from
        a grayscale input (imgIn(.
    scaleFact: scales the image by this factor prior
        to performing the analysis
    sBlur = size of the characteristic noise you want
        to remove with a gaussian blur.
    sAmount = magnitude of the unsharp mask used to process the image
    lnoise = size of the typical noise when segmenting the image
    lobject = typical size of the pbject to be considered
    boxSize = size of the box used to compute the intensity threshold
    solidThres = size of the lowest allowed cell solidity
    lengthThres = length of the cell at which you expect cell
        to start dividing (1.5 = 1.5 times the average length)
    widthThres = width of the cells. Cells above that width will be
        segmented as they may represent joined cells
    '''
    img = mat2gray(imgIn, 255)
    img = cv.resize(img, (np.size(imgIn, 1)*scaleFact,
                    np.size(imgIn, 0)*scaleFact))
    imgU = (unsharp(img, sBlur, sAmount))
    imgB = unsharp(bpass(imgU, lnoise, lobject).copy(), lnoise, 0)
    bwImg = cv.adaptiveThreshold(np.uint8(imgB), 255,
                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, int(boxSize), 0)

    #Dilate cells while maintaining them unconnected
    bwImg = dilateConnected(bwImg, 1)
    bwImg = segmentCells(bwImg > 0)
    bwImg = fragmentCells(bwImg > 0, regionprops(bwImg)[:, 6], solidThres)
    bwImg = removeSmallBlobs(bwImg, 50)
    bwImg = dilateConnected(bwImg, 1)
    return bwImg


def preProcessCyano(brightImg, chlorophyllImg=0, mask=False):
    '''
    Pre-process the Chlorophyll and brightfield images so that they can be
    segmented directly.
    '''
    if mask:
        bImg = bpass(chlorophyllImg, 1, 10)
        cellMask = cv.dilate(np.uint8(bImg > 50), None, iterations=15)
    else:
        cellMask = brightImg*0 + 1
    brightScaled = mat2gray(bpass(brightImg, 1, 10), 255)
    processedBrightfield = cv.adaptiveThreshold(np.uint8(brightScaled), 255,
                                                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv.THRESH_BINARY, 501, 0)
    dilatedIm = removeSmallBlobs(processedBrightfield*cellMask, 125)
#   dilatedIm = (removeSmallBlobs(processedBrightfield*cellMask, 15))
    dilatedIm[:10, :] = 255
    dilatedIm[-10:, :] = 255
    dilatedIm[:, :10] = 255
    dilatedIm[:, -10:] = 255
    bwL = bwlabel(dilatedIm)
    rP = regionprops(dilatedIm)
    idMax = (rP[:, 5] == rP[:, 5].max()).nonzero()[0][0] + 1
    dilatedIm[bwL == idMax] = 255

    imgOut = 255-dilatedIm.copy()

    return np.uint8(imgOut)


def peakdet(v,  delta,  x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Found here: https://gist.github.com/endolith/250860
    Returns two arrays

    function [maxtab,  mintab] = peakdet(v,  delta,  x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB,  MINTAB] = PEAKDET(V,  DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V,  and column 2 the found values.
    %
    %        With [MAXTAB,  MINTAB] = PEAKDET(V,  DELTA,  X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value,  and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer,  3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn,  mx = np.Inf,  -np.Inf
    mnpos,  mxpos = np.NaN,  np.NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos,  mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos,  mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return np.array(maxtab),  np.array(mintab)


def phaseLabelling(dataIn):

    Lin = dataIn[:]

    intLabel = np.zeros((len(Lin), ))

    if len(Lin) <= 12:
        return intLabel

    L1 = removePeaks(Lin, 'Down')

    Ls = smooth(L1, 10)

    pkMax, pkMin = peakdet(Ls, 7)
    if not len(pkMax) or not len(pkMin):
        return intLabel
    if pkMax[0, 0] < pkMin[0, 0]:
        pk1 = pkMax.copy()
        pk2 = pkMin.copy()
    elif pkMax[0, 0] > pkMin[0, 0]:
        pk1 = pkMin.copy()
        pk2 = pkMax.copy()

    regionList = 0.

    for pt in range(len(pk1)-1):
        minPos = (pk1[pt, 0] < pk2[:, 0]) & (pk1[pt + 1, 0] > pk2[:, 0])
        if np.sum(minPos):
            regionList = np.hstack((regionList, pk1[pt, 0]))
            minPos = pk2[minPos, 0][0]
            regionList = np.hstack((regionList,
                                    int(minPos/2. + pk1[pt, 0]/2.)))
            regionList = np.hstack((regionList,
                                    int(minPos)))
            regionList = np.hstack((regionList,
                                    int(minPos/2. + pk1[pt + 1, 0]/2.)))
            regionList = np.hstack((regionList, pk1[pt + 1, 0]))
    if pk2[-1, 0] > pk1[-1, 0]:
        regionList = np.hstack((regionList, int(pk1[-1, 0]/2.+pk2[-1, 0]/2.)))
        regionList = np.hstack((regionList, pk2[-1, 0]))

    dataOut = np.unique(regionList[1:]).astype('int')
    dataOut = np.vstack((dataOut, Ls[dataOut]))

#   timeArray = Tin[dataOut[:, 0].astype('int')]
    maxPt, minPt = peakdet(dataOut[1], 0.5)
    if not len(maxPt) or not len(minPt):
        return intLabel
    firstMin = int(minPt[0, 0])
    k = 0
    for id in range(firstMin, len(dataOut[0])-1):
        intLabel[dataOut[0, id]:dataOut[0, (id + 1)]] = k
        k = k + 1
        k = k % 4
    intLabel[int(dataOut[0, (id + 1)]):] = k
    k = 3
    for id in range(firstMin, 0, -1):
        intLabel[dataOut[0, id-1]:dataOut[0, id]] = k
        k = k - 1
        k = k % 4
    intLabel[0:int(dataOut[0, id-1])] = k

#   intLabel = gradualIntensity(intLabel)

    return np.transpose(intLabel)


def phaseLabelAll(trIn, dim=8):
    trI = splitIntoList(trIn, 4)

    for id in range(len(trI)):
        if len(trI[id]):
            intensityLabel = phaseLabelling(trI[id][:, dim])
            intensityLabel = gradualIntensity(intensityLabel)
            trI[id] = np.hstack((trI[id], np.zeros((len(trI[id]), 1))))
            if np.std(intensityLabel) != 0:
                trI[id][:, -1] = intensityLabel
            else:
                trI[id][:, -1] = intensityLabel*0-1

    trOut = revertListIntoArray(trI)
    return trOut


def gradualIntensity(intIn):
    '''
    intOut = gradualIntensity(intIn)
    '''
    intOut = np.double(intIn.copy()*0.)

    transitionPos = (np.diff(intIn) != 0).nonzero()[0]
    if len(transitionPos) == 0:
        return intIn
    val0 = intIn[0] - 1
    val1 = intIn[transitionPos[0]]

    if val0 == 3:
        val0 = -1

    gradInt = np.double(np.linspace(val0, val1, transitionPos[0]))
    intOut[:transitionPos[0]] = gradInt

    for i in range(1, len(transitionPos)):
        startPos = int(transitionPos[i-1])
        endPos = int(transitionPos[i]) + 1
        val0 = intIn[startPos]
        val1 = intIn[endPos-1]
        if val0 == 3:
            val0 = -1
        gradInt = np.double(np.linspace(val0, val1, endPos-startPos))
        intOut[startPos:endPos] = gradInt

    val0 = intIn[endPos]-1
    val1 = intIn[-1]

    if val0 == 3:
        val0 = -1
    gradInt = np.double(np.linspace(val0, val1, len(intOut)-endPos))
    intOut[endPos:] = gradInt

    return intOut + 1


def stabilizeImages(fPath, endsWith='tif', SAVE=True, preProcess=False):
    '''
    translationList = stabilizeImages(fPath, Save = True) goes
    through the files in folder fPath and aligns images by
    computing the optical flow between successive images
    '''
    tifList = []
    translationList = np.zeros((2, ), dtype=int)
    fileList = sort_nicely(os.listdir(fPath))
    for file in fileList:
        if file.endswith(endsWith):
            tifList.append(file)

    k = 0
    borderSize = 0
    img0 = 0
    fname0 = 0
    for fname in np.transpose(tifList):
        k = k + 1
        img = cv.imread(fPath + fname, -1)

        if k > 1:

            if preProcess:
                imgA = np.uint8(bpass(mat2gray(img0, 255), 1, 10))
                imgB = np.uint8(bpass(mat2gray(img[borderSize:-borderSize,
                                                   borderSize:-borderSize],
                                               255), 1, 10))
                A = (cv.matchTemplate(imgA, imgB, cv.cv.CV_TM_CCORR_NORMED))
            else:
                imgA = np.uint8(mat2gray(img0, 255))
                imgB = np.uint8(mat2gray(img[borderSize:-borderSize,
                                             borderSize:-borderSize], 255))
                A = (cv.matchTemplate(imgA, imgB, cv.cv.CV_TM_CCORR_NORMED))

            maxLoc = (A == A.max()).nonzero()

            deltaX = np.arange(-borderSize, borderSize + 1)[maxLoc[0]][0]
            deltaY = np.arange(-borderSize, borderSize + 1)[maxLoc[1]][0]

            if abs(deltaX) != 0:
                img = np.roll(img, int(deltaX), axis=0)
            if abs(deltaY) != 0:
                img = np.roll(img, int(deltaY), axis=1)
            if SAVE:
                cv.imwrite(fPath + fname0[:-10] + '-aligned.tif', img)
            translationList = np.vstack((translationList,
                                        np.array((int(deltaX), int(deltaY)))))
        else:
            borderSize = np.round(min(img.shape)/20.) - 1
        img0 = img.copy()
        fname0 = fname.copy()
    return translationList


def trackCells(fPath, lnoise=1, lobject=8, boxSize=15,
               lims=np.array([[0, -1]]), maxFiles=0, tList=0):
    '''
    This prepares the data necessary to track the cells.
    masterL, LL, AA = trackCells(fPath, lnoise = 1, lobject = 8,
                                 boxSize = 15, lims = 0)
    processes the files in fPath.

    lims is a Nx2 list containing the limits of each subregions.
    '''
    #Initialize variables
    sBlur = 0.5
    sAmount = 0
    scaleFact = 1
    solidThres = 0.65
    tifList = []
    masterList = []
    LL = []
    AA = []
    fileNum0 = 0
    fileNum = 0
    img = 0
    #Import and process image file
    fileList = sort_nicely(os.listdir(fPath))
    for file in fileList:
        if file.endswith('tif'):
            tifList.append(file)
    k = 0
    if maxFiles > 0:
        tifList = tifList[:maxFiles]
    kmax = len(tifList)
    startProgress('Analyzing files:')
    regionP = list(range(len(lims)))
    areaList = list(range(len(lims)))
    linkList = list(range(len(lims)))
    masterList = list(range(len(lims)))
    AA = list(range(len(lims)))
    LL = list(range(len(lims)))
    imgCropped = list(range(len(lims)))
    bwImg = list(range(len(lims)))
    bwL = list(range(len(lims)))
    bwL0 = list(range(len(lims)))
    for fname in np.transpose(tifList):
        print(fPath + fname)
        k = k + 1
        progress(np.double(k)/np.double(kmax))
        img0 = cv.imread(fPath + fname, -1)
        if len(img0) == 0:
            img = img.copy()
        else:
            img = img0.copy()
        if tList:
            img = np.roll(img, int(tList[k][0]), axis=0)
            img = np.roll(img, int(tList[k][1]), axis=1)
        img = cv.transpose(img)
        for id in range(len(lims)):
            imgCropped[id] = img[lims[id, 0]:lims[id, 1], :]
            bwImg[id] = processImage(imgCropped[id], scaleFact, sBlur,
                                     sAmount, lnoise, lobject,
                                     boxSize, solidThres)
            bwL[id] = bwlabel(bwImg[id])
            fileNum = k - 1
            if bwL[id].max() > 5:

                regionP[id] = regionprops(bwImg[id], scaleFact)
                avgCellI = avgCellInt(imgCropped[id].copy(), bwImg[id].copy())
                if np.isnan(avgCellI).any():
                    avgCellI[np.isnan(avgCellI)] = 0
                regionP[id] = np.hstack([regionP[id], avgCellI[1:]])
                if (fileNum-fileNum0) == 1:
                    areaList[id] = labelOverlap(bwL0[id], bwL[id])
                    AA[id].append(areaList[id])
                    linkList[id] = matchIndices(areaList[id], 'Area')
                    LL[id].append(linkList[id])
                #Extract regionprops
                if fileNum0 == 0:
                    masterList[id] = [regionP[id]]
                    AA[id] = []
                    LL[id] = []
                else:
                    masterList[id].append(regionP[id])
                bwL0[id] = bwL[id].copy()

        fileNum0 = fileNum + 0
    endProgress()
    return masterList, LL, AA


def updateTR(M, L, time):
    '''
    trTemp = updateTR(M, L, time)
    This function orders the values in masterL (M) from L at t = time.
    '''
    trTemp = np.zeros((len(L[time]), 8))
    parentIndex = (L[time][:, 0]-1).tolist()
    trT = M[time][parentIndex, :].take([0, 1, 8, 2, 3, 4, 7], axis=1)
    trTemp[:, [0, 1, 3, 4, 5, 6, 7]] = trT
    return trTemp


def linkTracks(masterL, LL):
    '''
    tr = linkTracks(masterL, LL) links the frame-to-frame tracks.
    It returns an array tr = [xPos, yPos, time, cellId, length, width,
                orientation, pixelIntensity, divisionEvents,
                familyID, age]
    '''
    #Links the frame-to-frame tracks

    totalTime = len(LL)
    ids = np.arange(len(LL[0])) + 1
    masterL[0] = np.hstack((masterL[0], np.zeros((len(masterL[0]), 1))))
    masterL[0][(LL[0][:, 0]-1).tolist(), 8] = ids
    tr = updateTR(masterL, LL, 0)
    tr[:, 2] = 0
    maxID = np.max(ids)
    tr0 = np.zeros((len(LL[1]), 8))
    startProgress('Linking cell overlaps:')

    for t in range(totalTime-1):
        progress(t/np.double(totalTime))
        tr0 = np.zeros((len(LL[t+1]), 8))
        masterL[t+1] = np.hstack([masterL[t+1],
                                  np.zeros((len(masterL[t+1]), 1))])
        masterL[t+1][(LL[t][:, 1]-1).tolist(), 8] = ids
        tr0 = updateTR(masterL, LL, t+1)
        tr0[:, 2] = t + 1
        ids = masterL[t+1][(LL[t+1][:, 0]-1).tolist(), 8]
        for i in range(len(ids)):
            if ids[i] == 0:
                maxID = maxID + 1
                ids[i] = maxID
                tr0[i, 3] = maxID
        tr = np.vstack([tr, tr0])
    tr = tr[tr[:, 2].argsort(-1, 'mergesort'), ]
    tr = tr[tr[:, 3].argsort(-1, 'mergesort'), ]
    endProgress()

    return tr


def processTracks(trIn, match=True):
    """
    tr = processTracks(trIn) returns a track with linked mother/daughter
    labels (tr[:, 8]),  identified family id (tr[:, 10]),  cell age (tr[:, 11])
    and the elongation rate (tr[:, 12])
    """
    #Find and assign the daughter ID at each division
    # (do it twice because the first time around is reorganizes the
    # trackIDs so that the mother is on the left).

    tr = trIn.copy()
    # 'Rematching the tracks'

    tr = rematchTracks(tr, 1.1, (tr[:, 1].max()+50, tr[:, 0].max()+100))

    # 'Fixing sudden cell size changes'
    tr = splitCells(tr)

    # "Bridging track gaps"
    tr = joinTracks(tr, 2.5, [2, 4])

    # "Splitting Tracks"
    tr = fixGaps(tr)
    tr = splitTracks(tr)

    # "Finding division events,  Merging tracks"
    tr = mergeTracks(tr)

    # "find family IDs"
    tr = findFamilyID(tr)

    if match:
        tr = matchFamilies(tr)

    # "fix family IDs"

    # "Adding cell Age"
    tr = addCellAge(tr)

    # "Adding Pole Age"
    tr = addPoleAge(tr)

    # "Compute elongation rate"
    tr = smoothDim(tr, 5)
    tr = addElongationRate(tr)

    return tr


def getBoundingBox(param, dist=1):
    '''
    boxPts = getBoundingBox(param, dist = 1)
    Return the polygon that bounds an ellipse
    defined by
    param = [xPos, yPos, time, id, length, orientation]
    The parameter dist denotes the factor by which the length
    and width are multiplied by when creating the bounding box.
    '''
    #Define parameters
    Pos = [param[0], param[1]]
    SemiMajor = dist*param[4]
    SemiMinor = dist*param[5]
    Orient = param[6]*np.pi/180

    RotMatrix = [[np.cos(Orient), -np.sin(Orient)],
                 [np.sin(Orient), np.cos(Orient)]]
    boxPts = [[SemiMinor/2, SemiMajor/2],
              [-SemiMinor/2, SemiMajor/2],
              [-SemiMinor/2, -SemiMajor/2],
              [SemiMinor/2, -SemiMajor/2],
              [SemiMinor/2, SemiMajor/2]]
    for elem in range(len(boxPts)):
        boxPts[elem] = np.dot(RotMatrix, boxPts[elem])+[Pos[0], Pos[1]]

    return np.array(boxPts)


def fixTracks(tr, dist, timeRange):
    '''
    listOfMatches = fixTracks(tr, dist, timeRange) finds the trackIDs
    that join track ends with nearby track starts.
    listOfMatches return [id1,  id2,  time,  distance]

    dist is the factor by which the box bounding a cells is
    multiplied by when looking for possible candidates.

    timeRange is given by [t1, t2] and denotes how far in the
    past (t1) and future (t2) should the script check to find
    the possible candidates.
    '''
    #Find track ends
    masterEndList = tr[np.diff(tr[:, 3]) > 0, :]

    #Find track starts
    masterStartList = tr[np.roll(np.diff(tr[:, 3]) > 0, 1), :]

    listOfMatches = matchTracksOverlap(masterEndList, masterStartList,
                                       dist, timeRange)

    return listOfMatches
    #Reassign the track IDs


def joinTracks(trIn, dist, dataRange):
    '''
    Merge the track ends with track starts. Do not care about cell divisions.
    '''

    mList0 = fixTracks(trIn, dist, dataRange)

    matchData = matchIndices(mList0, 'Area')
    trOut = mergeIndividualTracks(trIn.copy(), matchData.copy())

    return trOut


def removeShortCells(tr, shortCells=6):
    '''
    tr = removeShortTracks(tr, shortTrack = 3) removes
    tracks shorter than shortTrack
    '''

    shortIDs = np.unique(tr[tr[:, 4] < shortCells, 3])

    trI = splitIntoList(tr, 3)

    for i in shortIDs:
        trI[int(i)] = []

    trS = revertListIntoArray(trI)

    return trS


def removeShortTracks(tr, shortTrack=3, dim=3):
    '''
    tr = removeShortTracks(tr, shortTrack = 3) removes
    tracks shorter than shortTrack
    '''
    trLength = np.histogram(tr[:, dim], np.unique(tr[:, dim]))
    idShort = trLength[1][trLength[0] <= shortTrack]
    trT = splitIntoList(tr, dim)
        #Reassign cell ID
    for id in idShort:
        trT[int(id)] = []

    trS = revertListIntoArray(trT)
    return trS


def extendShortTracks(trIn, shortTrack=2):
    '''
    tr = removeShortTracks(tr, shortTrack = 3) removes
    tracks shorter than shortTrack
    '''
    tr = trIn.copy()
    trLength = np.histogram(tr[:, 3], np.arange(tr[:, 3].max()+10))
    idShort = trLength[1][trLength[0] <= shortTrack]
    trT = splitIntoList(tr, 3)
        #Reassign cell ID
    for id in idShort:
        if (id <= tr[:, 3].max()):
            if len(trT[int(id)]):
                trT[int(id)] = np.vstack((trT[int(id)], trT[int(id)][0]))
                trT[int(id)][-2, 2] = trT[int(id)][-1, 2] - 1

    trS = revertListIntoArray(trT)
    return trS


def matchTracksOverlap(inList1, inList2, dist, timeRange, orient=True):
    matchList = np.zeros((1, 3))
    startProgress('Matching tracks:')
    k = 0
    for pts in inList1:
        k += 1
        progress(np.double(k)/np.double(len(inList1)))
        boxPts = getBoundingBox(pts, dist*3)
        boxPath = pa.Path(boxPts.copy())

        #Find starting points that fall inside the bounding box
        nearPts = inList2[(inList2[:, 2] <= (pts[2]+timeRange[1])) &
                          (inList2[:, 2] > (pts[2]-timeRange[0])), :]

        centerPt = nearPts[:, 0:2]

        inIndx = boxPath.contains_points(centerPt)
        if sum(inIndx) > 0:
            inPts = nearPts[inIndx, :]
        else:
            inPts = []
        if len(inPts):
            xRange = np.array([inPts[:, 0].min()-100., inPts[:, 0].max()+100.])
            yRange = np.array([inPts[:, 1].min()-100., inPts[:, 1].max()+100.])

            pts[0] = pts[0]-xRange[0]
            pts[1] = pts[1]-yRange[0]

            inPts[:, 0] = inPts[:, 0]-xRange[0]
            inPts[:, 1] = inPts[:, 1]-yRange[0]

            dividingImg = traceOutlines([pts], dist,
                                        (np.floor(yRange[1]-yRange[0])+1,
                                         np.floor(xRange[1]-xRange[0])+1))
            areaList = np.zeros((1, 3))
            for id in range(len(inPts)):
                matchingImg = traceOutlines([inPts[id]], 1,
                                            (np.floor(yRange[1]-yRange[0])+1,
                                             np.floor(xRange[1]-xRange[0])+1))
                matchL = labelOverlap(dividingImg, matchingImg)
                matchL[:, 2] = matchL[:, 2]
                areaList = np.vstack((areaList, matchL))
        else:
            areaList = np.zeros((1, 3))
        if areaList.sum() == 0:
            areaList = np.zeros((1, 3))
        matchList = np.vstack((matchList, areaList))
#   matchList = matchList[matchList[:, 0] != matchList[:, 1], :]
    matchList = matchList[matchList[:, 2] > 0, :]
    endProgress()
    return matchList


def rematchTracks(trIn, dist, dim):
    '''
    Check if the correct cells are matched between two frames.
    '''

    inList = splitIntoList(trIn, 2)
    matchList = np.zeros((1, 3))
    startProgress('Rematching tracks:')
    nextImg = traceOutlines(inList[0], dist, (dim[0], dim[1]))
    idSwitch = np.zeros((1, 3))
    for k in range(len(inList)-1):
        progress(np.double(k)/np.double(len(inList)))

        #Find starting points that fall inside the bounding box
        currentImg = nextImg.copy()
        nextImg = traceOutlines(inList[k+1], dist, (dim[0], dim[1]))
        matchList = labelOverlap(currentImg, nextImg)
        if matchList.sum() == 0:
            matchList = np.zeros((1, 3))
        matchList = matchList[matchList[:, 2] > 0, :]

        newMatchList = matchIndices(matchList, 'Area')

        if np.sum(newMatchList[:, 0] != newMatchList[:, 1]) > 0:
            misMatch = newMatchList[newMatchList[:, 0] !=
                                    newMatchList[:, 1], :]
            if misMatch.shape[0] == 1:
                idSwitch = np.vstack((idSwitch,
                                      np.array((misMatch[0, 0],
                                                misMatch[0, 1], k))))
            else:
                for m in misMatch:
                    idSwitch = np.vstack((idSwitch, np.array((m[0], m[1], k))))

    idSwitch = idSwitch[1:]
    endProgress()

    trI = splitIntoList(trIn, 3)

    idT = idSwitch.copy()

    for k in np.unique(idT[:, 2]):
        x = idT[idT[:, 2] == k, :].copy()
        xt = idT.copy()
        if x.shape[0] == 1:
            idT[(xt[:, 0] == x[0, 0]) & (xt[:, 2] > x[0, 2]), 0] = x[0, 1]
            idT[(xt[:, 0] == x[0, 1]) & (xt[:, 2] > x[0, 2]), 0] = x[0, 0]
        elif x.shape[0] > 1:
            intersectID = np.intersect1d(x[:, 0], x[:, 1])
            if len(intersectID) > 0:
                for i in intersectID:
                    x[(x[:, 0] != i) & (x[:, 1] != i), :] = 0
            for i in range(x.shape[0]):
                idT[(xt[:, 0] == x[i, 0]) & (xt[:, 2] > x[i, 2]), 0] = x[i, 1]
                idT[(xt[:, 0] == x[i, 1]) & (xt[:, 2] > x[i, 2]), 0] = x[i, 0]

    for k in range(len(idSwitch)-1):
        x = idT[k]
        y = idSwitch[k]
        if x[0] != 0:
            trI[int(y[1])][trI[int(y[1])][:, 2] > x[2], 3] = x[0]
            trI[int(y[0])][trI[int(y[0])][:, 2] > x[2], 3] = x[1]

    trOut = revertListIntoArray(trI)

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]

    return trOut


def splitIntoList(listIn, dim):
    '''
    listOut = splitIntoList(listIn, dim)
    This function splits an array according to a specific index.
    For instance,  if dim = 1 is the time label,  it will create a list
    where list[t] is the data at time t.
    '''
    #Declare how fine the array will be split into
    divL = 100
    #Initialize the first slice
    listOut = list(range(1+int(np.max(listIn[:, dim]))))
    listT = listIn[listIn[:, dim] <= divL, :]
    for id in np.arange(len(listOut)):
        if listT.shape[0] > 0:
            listOut[int(id)] = (listT[listT[:, dim] == id, :])
        else:
            listOut[int(id)] = []
        if np.mod(id, divL) == 0:
            listT = listIn[(listIn[:, dim] > id) &
                           (listIn[:, dim] <= (id+divL)), :]

    return listOut


def revertListIntoArray(listIn):
    '''
    arrayOut = revertListIntoArray(listIn, dim)
    This function is the reverse of splitIntoList: it takes
    a list and concatenate the results into an array.
    '''
    #Declare the rough size of the array
    nID = 100

    arrayOut = list(range(int(np.floor(len(listIn)/nID)+2)))
    k = 0
    while len(listIn[k]) <= 0:
        k += 1
#   arrayOut[0] = listIn[k].copy()

    k = 0
    for id in range(len(listIn)):
        if (np.size(arrayOut[k]) == 1)and(np.size(listIn[id]) > 1):
            arrayOut[k] = listIn[id]
        elif isinstance(listIn[id], int):
            pass
        elif len(listIn[id]) > 0:
            arrayOut[k] = np.vstack((arrayOut[k], listIn[id]))
        if np.mod(id, nID) == 0:
            k = k + 1
    m = 0
    stop = False
    while not stop:
        try:
            arrayOutAll = arrayOut[m].copy()*0
            stop = True
        except:
            m += 1
            stop = False
    for id in range(0, len(arrayOut)):
        if np.size(arrayOut[id]) > 1:
            arrayOutAll = np.vstack((arrayOutAll, arrayOut[id]))

    arrayOutAll = arrayOutAll[arrayOutAll[:, 0] > 0, :]

    return arrayOutAll


def findDivisionEvents(trIn):
    '''
    divE = findDivisionEvents(trIn) goes through each cell tracks and
    finds the times at which each divides.
    divE returns an array with divE = [cellID, time]
    '''
    trT = splitIntoList(trIn, 3)
    divE = np.zeros((1, 2))
    for tr in trT:
        if np.size(tr) > 1:
            if len(tr) > 10:
                tr0 = tr.copy()
                tr0[(np.diff(tr0[:, 2]) == 0), :] = 0
                tr0 = tr0[tr0[:, 0] > 0, :]
                divEvents = findDivs(tr0[:, 4])
                if divEvents.any():
                    divTimes = tr[divEvents.tolist(), 2]
                    divT = np.vstack([divTimes*0 + tr[0, 3],
                                      divTimes])
                    divE = np.append(divE, divT.transpose(), 0)
    divE = divE[1:len(divE), :]
    return divE


def findDivs(L):
    '''
    divTimes = findDivs(L) finds the index of the division events.
    L is the length of the cell
    '''
    divTimes = np.array([])
    divJumpSize = 25
    minSize = 50
        #Remove the points with a large positive derivative,  do this twice
    L = removePeaks(L.copy(), 'Down')
    #L = removePlateau(L.copy())
    #Find the general location of a division event.
    divLoc = (np.diff((L)) < -divJumpSize).nonzero()

    #Check if length is higher later
    divTimes = np.array(divLoc[0])
    for i in range(len(divTimes)):
        divs = divTimes[i]
        if L[divs:(divs+3)].max() > L[divs]:
            divTimes[i] = 0
        elif L[divs] < minSize:
            divTimes[i] = 0

    divTimes = divLoc[0] + 1.
    divTimes[np.diff(divTimes) == 1] = 0
    divTimes = divTimes[divTimes != 0]
    divTimes = divTimes[divTimes > 3]
    return divTimes.astype('int')


def removePeaks(Lin, mode='Down'):
    '''
    This function removes the peaks in a function.
    Peaks are defined as a sudden increase followed by a
    sudden decrease in value within 4 datapoints.
    '''
    divJumpSize = 10
    LL = Lin.copy()
    #Find location of sudden jumps (up or down)
    jumpIDup = (np.diff(LL) > divJumpSize).nonzero()[0]
    jumpIDdown = (np.diff(LL) < -divJumpSize).nonzero()[0]

    if mode == 'Down':
        for id in jumpIDdown:
            if ((id-jumpIDup) == -1).nonzero()[0].size:
                LL[id+1] = LL[id].copy()
            elif ((id-jumpIDup) == -2).nonzero()[0].size:
                LL[id+1] = LL[id].copy()
                LL[id+2] = LL[id].copy()
            elif ((id-jumpIDup) == -3).nonzero()[0].size:
                LL[id+1] = LL[id].copy()
                LL[id+2] = LL[id].copy()
                LL[id+3] = LL[id].copy()
        jumpIDup = (np.diff(LL) > divJumpSize).nonzero()[0]
        jumpIDdown = (np.diff(LL) < -divJumpSize).nonzero()[0]

        for id in jumpIDdown:
            if ((id-jumpIDup) == 1).nonzero()[0].size:
                LL[id] = LL[id+1].copy()
            elif ((id-jumpIDup) == 2).nonzero()[0].size:
                LL[id-1] = LL[id+1].copy()
                LL[id] = LL[id+1].copy()
            elif ((id-jumpIDup) == 3).nonzero()[0].size:
                LL[id-2] = LL[id+1].copy()
                LL[id-1] = LL[id+1].copy()
                LL[id] = LL[id+1].copy()

    elif mode == 'Up':
        for id in jumpIDup:
            if ((id-jumpIDdown) == -1).nonzero()[0].size:
                LL[id+1] = (LL[id]+LL[id+2])/2.
            elif ((id-jumpIDdown) == -2).nonzero()[0].size:
                LL[id+1] = 2*LL[id]/3.+LL[id+3]/3.
                LL[id+2] = LL[id]/3.+2*LL[id+3]/3.
            elif ((id-jumpIDdown) == -3).nonzero()[0].size:
                LL[id+1] = 3*LL[id]/4.+LL[id+4]/4.
                LL[id+2] = LL[id]/2.+LL[id+4]/2.
                LL[id+3] = LL[id]/4.+3*LL[id+4]/4.
                jumpIDup = (np.diff(LL) > divJumpSize).nonzero()[0]
                jumpIDdown = (np.diff(LL) < -divJumpSize).nonzero()[0]

                for id in jumpIDup:
                    if ((id-jumpIDdown) == 1).nonzero()[0].size:
                        LL[id] = (LL[id-1]+LL[id+1])/2.
                    elif ((id-jumpIDdown) == 2).nonzero()[0].size:
                        LL[id-1] = 2*LL[id-2]/3.+LL[id+1]/3.
                        LL[id] = LL[id-2]/3.+2*LL[id+1]/3.
                    elif ((id-jumpIDdown) == 3).nonzero()[0].size:
                        LL[id-2] = 3*LL[id-3]/4.+LL[id+1]/4.
                        LL[id-1] = LL[id-3]/2.+LL[id+1]/2.
                        LL[id] = LL[id-3]/4.+3*LL[id+1]/4.

    return LL


def removePlateau(Lin):
    '''
    This function removes the plateaux in a function.
    A plateau is
    defined as a sudden increase (decrease) in value
    followed by a sudden decrease (increase). It is a plateau if the function
    becomes continuous once a constant value is added (substracted)
    to the plateau.
    '''

    divJumpSize = 12.5
    LL = Lin.copy()

    #Find the location of the sudden jumps (up or down)
    jumpIDup = (np.diff(LL) > divJumpSize).nonzero()[0]
    jumpIDup = jumpIDup[jumpIDup > 0]
    jumpIDup = jumpIDup[jumpIDup < (len(LL)-1)]

    jumpIDdown = (np.diff(LL) < -divJumpSize).nonzero()[0]
    jumpIDdown = jumpIDdown[jumpIDdown > 0]
    jumpIDdown = jumpIDdown[jumpIDdown < (len(LL)-1)]

    nextDown = -1
    previousDown = -1

    for id in jumpIDup:
        if jumpIDdown[jumpIDdown > id].any():
            nextDown = jumpIDdown[jumpIDdown > id][0]
        if jumpIDdown[jumpIDdown <= id].any():
            previousDown = jumpIDdown[jumpIDdown <= id][-1]

        if nextDown > 0:
            jumpDiffDown = np.abs(np.abs(LL[id]-LL[id+1]) -
                                  np.abs(LL[nextDown]-LL[nextDown+1]))
        else:
            jumpDiffDown = np.Inf
        if previousDown >= -1:
            jumpDiffUp = np.abs(np.abs(LL[id]-LL[id+1]) -
                                np.abs(LL[previousDown]-LL[previousDown+1]))
        else:
            jumpDiffUp = np.Inf
        if jumpDiffDown < jumpDiffUp:
            LL[(id+1):(nextDown+1)] = LL[(id+1):(nextDown+1)]-(LL[id+1]-LL[id])
        elif jumpDiffUp < jumpDiffDown:
            LL[(previousDown+1):(id+1)] = LL[(previousDown+1):
                                             (id+1)]+(LL[id+1]-LL[id])
        nextDown = -10
        previousDown = -10
    return LL


def rolling_window(a,  window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1,  window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a,  shape=shape,  strides=strides)


def splitTracks(trIn):
    '''
    This function splits each tracks into cell segment that start with birth
    and ends with cell divisions or device escape.
    '''
    trIn = np.hstack((trIn, np.zeros((len(trIn), 2))))
    trI = splitIntoList(trIn, 3)
    divData = findDivisionEvents(trIn.copy())
    x = np.array([0, 0])
    k = 1
    idList = np.unique(trIn[:, 3]).astype('int')
    startProgress('Splitting cell tracks:')
    for id in idList:
        progress(np.double(id)/np.double(idList.max()))
        if np.intersect1d(divData[:, 0], np.array(id)):
            divList = divData[divData[:, 0] == id, :]
            x[0] = trI[id][0, 2]
            for pt in divList:
                x[1] = pt[1].copy()
                if x[1] < trI[id][-1, 2]:
                    if (pt[1] != divList[0][1]):
                        trI[id][(trI[id][:, 2] == x[0]), 8] = k - 1
                        trI[id][(trI[id][:, 2] == x[0]), 9] = k - 1
                    trI[id][(trI[id][:, 2] == (x[1]-1)), 8] = k + 1
                    trI[id][(trI[id][:, 2] >= x[0]) &
                            (trI[id][:, 2] < x[1]), 3] = k
                    k = k + 1
                    x[0] = x[1].copy()

            if sum(trI[id][:, 2] >= x[0]) > 1:
                trI[id][(trI[id][:, 2] == x[0]), 8] = k - 1
                trI[id][(trI[id][:, 2] == x[0]), 9] = k - 1
                trI[id][trI[id][:, 2] >= x[0], 3] = k
            if len(trI[id][trI[id][:, 2] >= x[0], 3]) == 1:
                trI[id][trI[id][:, 2] >= x[0], 3] = 0
#               trI[id] = np.vstack((trI[id], trI[id][-1, :]))
#               trI[id][-1, 2] = trI[id][-1, 2]+1
#               trI[id][-1, 8:10] = 0
            k = k + 1
        else:
            trI[id][:, 3] = k
            k = k + 1
    trOut = revertListIntoArray(trI)
    trOut = trOut[trOut[:, 3] > 0, :]
#   trOut = removeShortCells(trOut, 20)
    endProgress()
    trOut = extendShortTracks(trOut, 1)
    return trOut


def splitCells(trIn):
    '''
    trOut = splitCells(trIn) looks at each track segment and
    if it finds that a cell suddendly increases in length,
    it will split it into two small cells instead
    '''

    freeID = np.setdiff1d(np.arange(5*np.max(trIn[:, 3])),
                          np.unique(trIn[:, 3]))
    freeID = freeID[1:].copy()
    startProgress('Splitting joined cells: ')
    trI = splitIntoList(trIn, 3)
    for i in range(len(trI)):
        if len(trI[i]):
            progress(np.double(i)/np.double(len(trI)))
            L = trI[i][:, 4].copy()
            Lp = removePlateau(L)
            jumpID = ((L-Lp) > 0).nonzero()[0]
            endI = len(L)
            if jumpID.any():
                id0 = jumpID[0]
                for id in jumpID:
                    if (id > 3) & (id < (len(L)-3)):
                        if (id-id0) > 1:
                            freeID = freeID[2:]
                        trI[i] = np.vstack((trI[i], trI[i][id, :]))
                        ellN = trI[i][id, 4]/4.
                        angN = trI[i][id, 6]*np.pi/180.
                        yp = trI[i][id, 1] + ellN*np.cos(angN)
                        trI[i][id, 1] = yp
                        trI[i][id:endI, 4] = Lp[id:endI]
                        trI[i][id, 0] = trI[i][id, 0]-ellN*np.sin(angN)
                        trI[i][id:endI, 3] = freeID[0]

                        trI[i][-1, 3] = freeID[1]
                        trI[i][-1, 0] = trI[i][-1, 0] + ellN*np.sin(angN)
                        trI[i][-1:, 1] = trI[i][-1:, 1]-ellN*np.cos(angN)
                        trI[i][-1, 4] = trI[i][id, 4]

                        id0 = id + 0.
                freeID = freeID[2:]
            jumpIDDown = ((L-Lp) < 0).nonzero()[0]
            if jumpIDDown.any():
                for id in jumpIDDown:
                    if (id > 3) & (id < (len(L)-3)):
                        trI[i][id, 4] = Lp[id]
    trOut = revertListIntoArray(trI)

    trOut = extendShortTracks(trOut, 1)

    endProgress()

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]
    return trOut


def findDaughters(trIn, merge=False):
    '''

    '''
    tr = trIn.copy()

    #Find and match division events
    if len(trIn[0, :]) > 8:
        divData0 = tr[(tr[:, 8] > 0) & (tr[:, 9] == 0), :].take([3, 8], axis=1)
        divData0 = np.hstack((divData0, 1000*np.ones((len(divData0), 1))))

    else:
        divData0 = np.zeros((1, 3))
    tr0 = tr.copy()
    mList0 = fixTracks(tr0, 2.5, [2, 4])
    #if np.sum(mList0[:, 3] < meanL):
#       mList1 = mList0[mList0[:, 3] < meanL, :]
#   else:
    mList1 = mList0.copy()
    divData1 = matchIndices(mList1, 'Area')
    for dd in divData0:
        divData1[divData1[:, 1] == dd[1], :] = 0
        divData1[divData1[:, 0] == dd[0], :] = 0

    divData1 = divData1[divData1[:, 0] > 0, :]

    mList2 = mList0.copy()
    for div in divData0:
        mList2[(mList0[:, 1] == div[1]), :] = 0
    for div in divData1:
        mList2[(mList0[:, 1] == div[1]), :] = 0

    mList2 = mList2[mList2[:, 0] > 0, :]

    divData2 = matchIndices(mList2, 'Area')

    divData = np.vstack((divData0, divData1, divData2))
    divData = divData[divData[:, 0].argsort(), ]
    divData = unique_rows(divData)

    divData = removeImpossibleMatches(trIn, divData)

    return divData


def removeImpossibleMatches(trIn, divData0):
    '''
    divData = removeImpossibleMatches(trIn, divData) goes through each matching
    IDs and makes sure it does not create matching loops (ie. daughter
    gives birth to mother).
    '''

    divData = divData0.copy()

    trI = splitIntoList(trIn, 3)

    for k in range(len(divData)):
        dd = divData[k]
        motherID = dd[0]
        daughterID = dd[1]

        if trI[int(motherID)][-1, 2] >= trI[int(daughterID)][-1, 2]:
            divData[k, :] = 0

    divOut = divData[divData[:, 0] > 0, :]

    if not len(divOut):
        divOut = np.zeros((3, 3))

    return divOut


def mergeIndividualTracks(tr, divD):
    '''
    Merge tracks from the index list divData. Only merge if data is continuous
    or cell is larger in the near future.
    '''
    divData = divD.copy()
    trIn = tr.copy()
    trT = splitIntoList(trIn.copy(), 3)
    idList = np.unique(divData[:, 0])
    for k in range(len(idList)):
        motherCell = int(idList[k])
        daughterCell = divData[divData[:, 0] == motherCell, 1].astype('int')
        if len(daughterCell) == 1:
            trT[motherCell] = np.vstack((trT[motherCell], trT[daughterCell]))
            trT[motherCell][:, 3] = motherCell
            trT[daughterCell] = []
            idList[idList == daughterCell] = motherCell
            divData[divData[:, 0] == motherCell, 0] = 0
            divData[divData[:, 0] == daughterCell, 0] = motherCell
    trOut = revertListIntoArray(trT)

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]

    return trOut


def mergeTracks(trIn):
    '''
    trOut = mergeTracks(trIn) goes through each division events and attemps
    to match it with a mother cell.
    '''
    #Fix tracks that skip frames without divisions
    #trOut = removeShortTracks(trIn, 1)
    trOut = trIn.copy()
    idSwitch = np.array((0, 0))

    #Get division data
    divData = findDaughters(trOut)
    masterEndList = trOut[np.diff(trOut[:, 3]) > 0, :]
    for i in np.unique(masterEndList[:, 3]):
        dd = divData[divData[:, 0] == i, :]
        if len(dd) > 1:
            masterEndList[masterEndList[:, 3] == i, :] = 0
    masterEndList = masterEndList[masterEndList[:, 0] > 0, :]

    orphanCoordinates = np.zeros((1, trOut.shape[1]))

    for i in np.unique(trOut[trOut[:, 2] > 50, 3]):
        dd = divData[divData[:, 1] == i, :]
        if len(dd) < 1:
            orphanCoordinates = np.vstack((orphanCoordinates,
                                           trOut[trOut[:, 3] == i, :][0]))

    #Link between orphan tracks at birth and track ends
    mList = matchTracksOverlap(masterEndList, orphanCoordinates, 2, [3, 6])
    mList = mList[mList[:, 0] > 0, :]
    mList = mList[mList[:, 0] != mList[:, 1], :]
    if mList.any():
        matchData = matchIndices(mList, 'Area')
        matchData = removeImpossibleMatches(trIn, matchData)
    else:
        matchData = np.zeros((3, 2))

    trT = splitIntoList(trOut, 3)
    ids, indx = np.unique(trOut[trOut[:, 2].argsort(), 3], return_index=True)
    ids = ids[indx.argsort()]
    ids = np.flipud(ids)
    startProgress('Merging tracks:')
    for k in range(len(ids)):
        i = ids[k]
        progress(np.double(k)/np.double(len(ids)))
        divD = divData[(divData[:, 0] == i) & (divData[:, 1] != i), :]
        mData = matchData[(matchData[:, 0] == i) & (matchData[:, 1] != i), :]

        if len(divD):
            divD = divD[divD[:, 0] != divD[:, 1], :]
        if len(mData):
            mData = mData[mData[:, 0] != mData[:, 1], :]

        if len(divD) == 2:
            trT[int(i)][-1, 8] = divD[0, 1]
            trT[int(divD[0, 1])][0, 8] = i
            trT[int(divD[0, 1])][0, 9] = i
            trT[int(i)][-1, 9] = divD[1, 1]
            trT[int(divD[1, 1])][0, 8] = i
            trT[int(divD[1, 1])][0, 9] = i
        elif (len(mData) == 1)and(len(divD) == 1):
            trT[int(i)][-1, 8] = divD[0, 1]
            trT[int(divD[0, 1])][0, 8] = i
            trT[int(divD[0, 1])][0, 9] = i
            trT[int(i)][-1, 9] = mData[0, 1]
            trT[int(mData[0, 1])][0, 8] = i
            trT[int(mData[0, 1])][0, 9] = i
        elif (len(mData) == 1)and(len(divD) == 0):
            if checkContinuous(trT, int(i), int(mData[0, 1])):
                idSwitch = np.vstack((idSwitch, np.array((i, mData[0, 1]))))
            else:
                trT[int(i)][-1, 8] = mData[0, 1]
                trT[int(i)][-1, 9] = mData[0, 1]
                trT[int(mData[0, 1])][0, 8] = i
                trT[int(mData[0, 1])][0, 9] = i
        elif (len(mData) == 0)and(len(divD) == 1):
            if checkContinuous(trT, int(i), int(divD[0, 1])):
                idSwitch = np.vstack((idSwitch, np.array((i, divD[0, 1]))))
            else:
                trT[int(i)][-1, 8] = divD[0, 1]
                trT[int(i)][-1, 9] = divD[0, 1]
                trT[int(divD[0, 1])][0, 8] = i
                trT[int(divD[0, 1])][0, 9] = i

    trOut = revertListIntoArray(trT)

    for k in range(1, len(idSwitch[:, 0])):
        m = idSwitch[k, :]
        trOut[trOut[:, 3] == m[1], 8:10][0] = 0
        trOut[trOut[:, 3] == m[1], 3] = m[0]
        trOut[trOut[:, 8] == m[1], 8] = m[0]
        trOut[trOut[:, 9] == m[1], 9] = m[0]
        idSwitch[idSwitch[:, 0] == m[1], 0] = m[0]

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]

    trOut[(trOut[:, 8] > 0) & (trOut[:, 9] == 0), 8:10] = 0
    endProgress()

    return trOut


def checkContinuous(trIn, id1, id2):
    '''

    '''

    if trIn[id1][0, 2] < trIn[id2][0, 2]:
        dT1 = trIn[id1].copy()
        dT2 = trIn[id2].copy()
    else:
        dT2 = trIn[id1].copy()
        dT1 = trIn[id2].copy()

    L = np.hstack((dT1[:, 4], dT2[:, 4]))
    if len(L) > 10:
        divs = findDivs(L)
    else:
        return True
    if divs.any():
        return False
    else:
        return True


def findFamilyID(trIn):
    """
    Label the cells that are member of the same family with their
    family ID in the 10th column.
    """
    trIn = np.hstack([trIn, np.zeros((len(trIn), 1))])
    trI = splitIntoList(trIn.copy(), 3)
    cellIdList = np.unique(trIn[trIn[:, -1] == 0, 3])
    famID = 0
    stop = False
    idMax = len(cellIdList)
    startProgress('Finding and sorting the families:')
    while len(cellIdList) > 0:
        progress(1-np.double(len(cellIdList))/np.double(idMax))
        famID = famID + 1
        unlabelledCell = cellIdList[0]
        famList = np.array([unlabelledCell])
        while not stop:
            famList1 = famList.copy()
            for id in famList:
                if len(trI[int(id)]) > 1:
                    famList0 = np.unique(trI[int(id)][-1, 8:10])
                    famList = np.unique(np.hstack([famList,
                                                   famList0]))
            if np.array_equal(famList, famList1):
                stop = True
        famList = famList[famList > 0]
        for id in famList:
            trI[int(id)][:, -1] = famID
        cellIdList = np.setdiff1d(cellIdList, famList)

        stop = False
    trLong = revertListIntoArray(trI)
    endProgress()
    return trLong


def matchFamilies(trIn):
    '''
    This script matches the start of a new family with
    the closest cell.
    '''
    trIn = removeShortTracks(trIn, 50, 10)
    freeID = np.setdiff1d(np.arange(2*np.max(trIn[:, 3])),
                          np.unique(trIn[:, 3]))
    freeID = freeID[1:].copy()
    trIn = np.vstack((trIn, trIn[0]))
    trIn[-1, :] = 0
    trIn[-1, 3] = 2*np.max(trIn[:, 3])

    trFam = splitIntoList(trIn, 10)
    trI = splitIntoList(trIn, 3)

    famStart = trIn[0]
    for tt in trFam:
        if len(tt):
            if len(tt[:, 0]) > 10:
                tt = tt[tt[:, 2].argsort(-1, 'mergesort'), ]
                famStart = np.vstack((famStart, tt[0]))

    initialIDs = np.unique(trIn[trIn[:, 2] < 25, 10])
    for i in initialIDs:
        famStart[famStart[:, 10] == i, :] = 0
    famStart = famStart[famStart[:, 0] > 0, :]

    trI2 = splitIntoList(trIn, 3)

    for i in range(len(trI2)):
        if len(trI2[i]) > 31:
            trI2[i] = trI2[i][15:-15, :].copy()
        else:
            trI2[i] = []

    trC = revertListIntoArray(trI2)

    listOfMatches = matchTracksOverlap(famStart.copy(), trC.copy(), 5, [5, -1])
    if len(listOfMatches):
        matchData = matchIndices(listOfMatches, 'Area')
    else:
        matchData = []
    for md in matchData:
        trI[int(md[0])][0, 8:10] = md[1]

        divTime = trI[int(md[0])][0, 2]
        if len(trI[int(md[1])][trI[int(md[1])][:, 2] < divTime, 3]) == 0:
            divTime += 2
        trI[int(md[1])][trI[int(md[1])][:, 2] >= divTime, 3] = freeID[0]
        if len(trI[int(md[1])][trI[int(md[1])][:, 2] >= divTime, 3]) == 1:
            trI[int(md[1])] = np.vstack((trI[int(md[1])],
                                         trI[int(md[1])][-1, :]))
            trI[int(md[1])][-2, 8:10] = 0
            trI[int(md[1])][-1, 2] = trI[int(md[1])][-1, 2] + 1
        trI[int(md[1])][trI[int(md[1])][:, 2] == (divTime), 8] = md[1]
        trI[int(md[1])][trI[int(md[1])][:, 2] == (divTime), 9] = md[1]
        trI[int(md[1])][trI[int(md[1])][:, 2] == (divTime-1), 8] = md[0]
        trI[int(md[1])][trI[int(md[1])][:, 2] == (divTime-1), 9] = freeID[0]

        daughterID1 = trI[int(md[1])][-1, 8]
        daughterID2 = trI[int(md[1])][-1, 9]
        if (daughterID1 > 0) & (daughterID1 <= len(trI)):
            if len(trI[int(daughterID1)]):
                trI[int(daughterID1)][0, 8:10] = freeID[0]
        if (daughterID2 > 0) & (daughterID2 <= len(trI)):
            if len(trI[int(daughterID2)]):
                trI[int(daughterID2)][0, 8:10] = freeID[0]

        freeID = freeID[1:].copy()

    trOut = revertListIntoArray(trI)

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]
    trOut = findFamilyID(trOut)

    trOut[:, 10] = trOut[:, 11].copy()

    trOut = trOut[:, :11].copy()

    trF = splitIntoList(trOut, 10)

    for i in range(len(trF)):
        if len(np.unique(trF[i][:, 3])) <= 1:
            trF[i] = []

    trOut = revertListIntoArray(trF)

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]

    return trOut


def fixGaps(trIn):
    '''
    Adds the age of the cells in the last column.
    It also fills in the blank and missing data (eg. when track skip a time)
    '''

    trI = splitIntoList(trIn, 3)

    startProgress('Fixing the gaps in each tracks:')
    for i in range(len(trI)):
        progress(i/np.double(len(trI)))
        if len(trI[i]) > 0:
            if (np.diff(trI[i][:, 2]) > 1).any():
                trI[i] = fillGap(trI[i].copy(), age=False)
    trOut = revertListIntoArray(trI)
    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] == 0
    trOut = trOut[trOut[:, 0] > 0, :]

    endProgress()
    return trOut


def addCellAge(trIn):
    '''
    Adds the age of the cells in the last column.
    It also fills in the blank and missing data (eg. when track skip a time)
    '''

    trIn = np.hstack((trIn, np.zeros((len(trIn), 1))))
    trI = splitIntoList(trIn, 3)

    for i in range(len(trI)):
        if len(trI[i]) > 0:
            motherID = trI[i][0, 8]
            if motherID > 0:
                if len(trI[int(motherID)]) > 0:
                    divisionTime = trI[int(motherID)][-1, 2]
            else:
                divisionTime = trI[i][0, 2]
            tStart = trI[i][0, 2]-divisionTime-1
            trI[i][0, -1] = np.max((tStart, 0))
            trI[i][1:, -1] = trI[i][0, -1] + np.cumsum(np.diff(trI[i][:, 2]))
            if trI[i][0, -1] > 0:
                trI[i] = np.vstack((-1*np.ones((tStart, trI[i].shape[1])),
                                   trI[i]))
                trI[i][:tStart, 0] = trI[i][tStart, 0]
                trI[i][:tStart, 1] = trI[i][tStart, 1]
                trI[i][:tStart, 2] = trI[i][tStart, 2]-np.arange(tStart, 0, -1)
                trI[i][:, 3] = i
                trI[i][:tStart, 4] = trI[i][tStart, 4]
                trI[i][:tStart, 5] = trI[i][tStart, 5]
                trI[i][:tStart, 6] = trI[i][tStart, 6]
                trI[i][:tStart, 7] = trI[i][tStart, 7]
                trI[i][0, 8] = trI[i][tStart, 8]
                trI[i][0, 9] = trI[i][tStart, 9]
                trI[i][1:-1, 8] = 0
                trI[i][1:-1, 9] = 0
                trI[i][:, 10] = trI[i][-1, 10]
                trI[i][:tStart, -1] = np.arange(tStart)

            if (np.diff(trI[i][:, 2]) > 1).any():
                trI[i] = fillGap(trI[i].copy())
    trOut = revertListIntoArray(trI)

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 2]) == 0) & (np.diff(trOut[:, 3]) == 0), :] == 0
    trOut = trOut[trOut[:, 0] > 0, :]

    return trOut


def fillGap(tr, age=True):
    '''

    '''
    trIn = tr.copy()
    stop = False
    k = 0
    while stop is False:
        dt = trIn[k+1, 2]-trIn[k, 2]
        for i in range(int(dt-1)):
            dTemp = trIn[k, :].copy()
            dTemp[2] = trIn[k, 2] + 1
            if age:
                dTemp[11] = trIn[k, 11] + 1
            trIn = np.insert(trIn, k + 1, dTemp, 0)
            k = k + 1
        if (k+2) == len(trIn):
            stop = True
        else:
            k = k + 1

    return trIn


def addElongationRate(trIn):
    '''
    Fits an exponential to the length between every divisions to find the
    elongation rate.
    '''
    trIn = np.hstack((trIn, np.zeros((len(trIn), 1))))
    trI = splitIntoList(trIn, 3)

    for id in np.unique(trIn[:, 3]):
        dT = trI[int(id)].copy()
        if len(dT) > 15:
            dT = dT[5:-5, :]
            z = np.polyfit(range(len(dT)),  np.log(np.abs(dT[:, 5])),  1)
            dT[:, -1] = z[0]
            trI[int(id)][:, -1] = z[0]
    trOut = revertListIntoArray(trI)

    return trOut


def smoothDim(trIn, dim):
    '''
    Goes through each track and applies a removePeaks to the length
    '''

    trI = splitIntoList(trIn, 3)

    for id in range(len(trI)):
        dT = trI[id]
        if len(dT):
            LL = removePeaks(dT[:, dim], 'Down')
            if (dim == 4) or (dim == 5):
                LL = removePlateau(LL)
            trI[id][:, dim] = LL.copy()

    trOut = revertListIntoArray(trI)
    return trOut


def extractLineage(trI, id, dim=8):
    '''
    lin = extrackLineage(trIn, id) return the lineage starting from id .
        The data is ordered chronologically as lin[n] = [motherID,  time]
    '''
    try:
        trI = splitIntoList(trI, 3)
    except:
        pass

    t = trI[id][0, 2]
    lin = np.array((id, t))
    while t > 0:
        motherID = trI[id][0, dim]
        if motherID == trI[id][0, 3]:
            t = 0
        elif motherID > 0:
            t = trI[int(motherID)][0, 2]
            lin = np.vstack((lin, np.array((motherID, t))))
            id = int(motherID)
        else:
            t = 0
        if trI[id].shape[0] == 1:
            t = 0
    if lin.ndim > 1:
        lin = np.flipud(lin)
    return lin


def findPoleAge(trIn, id, t):
    '''
    [ageL, ageR] = findPoleAge(trIn, id, t) return the age of the pole of a
    cell ageL is the age of the leftmost pole and ageR is the age of the
    rightmost pole
    '''

    if t > trIn[id][-1, 2]:
        print "make sure t is within id's lifetime"
        return np.zeros((2, ))
    elif t < trIn[id][0, 2]:
        print "make sure t is within id's lifetime"
        return np.zeros((2, ))

    lin = extractLineage(trIn, id)
    if not len(lin):
        return np.array((0, 0))
    if lin.ndim == 1:
        return np.array((t-trIn[int(id)][0, 2], t-trIn[int(id)][0, 2]))

    ageL = 0
    ageR = 0

    dTD = trIn[int(id)]
    for k in range(1, len(lin[:, 0])):
        sisterIDs = trIn[int(lin[k-1, 0])][-1, 8:10]

        dTM = trIn[int(lin[k-1, 0])]
        dTD = trIn[int(lin[k, 0])]
        if sisterIDs[0] == sisterIDs[1]:
            if ageL >= ageR:
                ageL += lin[k, 1]-lin[k-1, 1]
                ageR = 0
            else:
                ageL = 0
                ageR += lin[k, 1]-lin[k-1, 1]
        else:
            dTS = trIn[int(sisterIDs[sisterIDs != lin[k, 0]])]
            bbM = getBoundingBox(dTM[-1, :])
            bbD = getBoundingBox(dTD[0, :])
            bbS = getBoundingBox(dTS[0, :])

            hypotArray = np.hypot((bbD-bbM)[:, 0], (bbD-bbM)[:, 1])
            hypotArrayS = np.hypot((bbS-bbM)[:, 0], (bbD-bbM)[:, 1])
            if hypotArray[0] < hypotArray[2]:
                if hypotArray[0] <= hypotArrayS[0]:
                    ageL += lin[k, 1]-lin[k-1, 1]
                    ageR = 0
                else:
                    ageL = 0
                    ageR += lin[k, 1]-lin[k-1, 1]
            elif hypotArray[0] > hypotArray[2]:
                if hypotArray[0] > hypotArrayS[0]:
                    ageL = 0
                    ageR += lin[k, 1]-lin[k-1, 1]
                else:
                    ageL += lin[k, 1]-lin[k-1, 1]
                    ageR = 0
    ageL += t-dTD[0, 2]
    ageR += t-dTD[0, 2]
    return np.array((ageL, ageR))


def addPoleAge(trIn):
    '''
    trOut = addPoleAge(trIn) adds two columns to trIn,  each
    one containing the left or the right pole age.
    '''

    trI = splitIntoList(trIn, 3)
    startProgress('Adding pole age:')

    for k in range(len(trI)):
        if len(trI[k]):
            progress(np.double(k)/np.double(len(trI)))
            poleAges = findPoleAge(trI, int(trI[k][0, 3]), trI[k][0, 2])

            trI[k] = np.hstack((trI[k], np.zeros((len(trI[k]), 2))))

            if poleAges[0] == poleAges[1]:
                age1 = trI[k][:, 11] + poleAges[0]
                age2 = trI[k][:, 11] + 0
            else:
                age1 = trI[k][:, 11] + poleAges[0]
                age2 = trI[k][:, 11] + poleAges[1]
            trI[k][:, -2] = age1
            trI[k][:, -1] = age2
            if trI[k][0, -1] == 0:
                trI[k][:, -2] = age2
                trI[k][:, -1] = age1
    trOut = revertListIntoArray(trI)

    trOut[:, -2] = trOut[:, -1]
    endProgress()

    trOut = addPoleID(trOut[:, :-1])

    return trOut


def addPoleID(trIn):
    '''
    trOut = addPoleID(trIn) returns a track array with an additional ID at
    position trOut[:, 5] that tracks the oldest pole. New poles are assigned a
    new ID. All other columns are shifted to the right.
    '''

    trIn = np.hstack((trIn, np.zeros((len(trIn[:, 0]), 1))))

    trI = splitIntoList(trIn, 3)

    poleID = 0
    ids, indx = np.unique(trIn[trIn[:, 2].argsort(), 3], return_index=True)
    ids = ids[indx.argsort()]
#   ids = np.flipud(ids)
    startProgress('Adding pole ID:')
    k = 0
    kMax = len(ids)
    for id in ids.astype('int'):
        k += 1
        if len(trI[id]):
            progress(np.double(k)/np.double(kMax))
            dID1 = trI[id][-1, 8]
            dID2 = trI[id][-1, 9]
            if trI[id][0, -1] == 0:
                poleID += 1
                pID = poleID
            else:
                pID = trI[id][0, -1]

            if (dID1 > 1):
                if trI[int(dID1)][0, 12] >= trI[int(dID2)][0, 12]:
                    dID = dID1
                    trI[int(dID)][:, -1] = pID
                elif trI[int(dID1)][0, 12] < trI[int(dID2)][0, 12]:
                    dID = dID2
                    trI[int(dID)][:, -1] = pID
                trI[id][:, -1] = pID
            else:
                trI[id][:, -1] = pID

    trOut = revertListIntoArray(trI)
    endProgress()
    trT = trOut[:, [0, 1, 2, 3, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    trOut[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = trT

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 4].argsort(-1, 'mergesort'), ]

    trOut[(np.diff(trOut[:, 4]) == 0) &
          (np.diff(trOut[:, 2]) == 0) &
          (trOut[:-1, 9] == 0), :] = 0
    trOut = trOut[trOut[:, 0] > 0, :]

    trOut = trOut[trOut[:, 2].argsort(-1, 'mergesort'), ]
    trOut = trOut[trOut[:, 3].argsort(-1, 'mergesort'), ]

    return trOut


'''

def computeOpticalFlow:
    for i in range(800):
        bar0 = array(im)
        im.seek(i)
            bar1 = array(im)
        A = cv2.calcOpticalFlowFarneback(uint16(bar0), uint16(bar1),
                         None, pyr_scale = 0.5, levels = 1,
                         winsize = 25, iterations = 1,
                         poly_n = 5, poly_sigma = 1.1,
                         flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        xx[i] = sum(A[0:700, :, 1])
        print i
'''

'''
    xx = zeros((600, 1))
    for i in range(350):
        bar0 = cv2.transpose(cv2.imread('Fluo_scan_'+str(i)+'_Pos0_g.tif', -1))
        bar1 = cv2.transpose(cv2.imread('Fluo_scan_'+str(i+1)+'_Pos0_g.tif',
                                        -1))
        A = cv2.calcOpticalFlowFarneback(uint8(bar0), uint8(bar1),
                     pyr_scale = 0.5, levels = 1, winsize = 25,
                     iterations = 1, poly_n = 5, poly_sigma = 1.1,
                     flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        xx[i] = sum(A[:, :, 0])
        print i
'''


def optimizeParameters(fPath, num):
    """
    Tests the processing parameters on a test file
    """
    import random as rnd
    lnoise0 = 2
    lobject0 = 8
    boxSize0 = 25
    tifList = []
    fileList = sort_nicely(os.listdir(fPath))
    for file in fileList:
        if file.endswith('tif'):
            tifList.append(file)
    if num == 0:
        num = rnd.randint(1, len(tifList))
        print num
    stop = False
    while not stop:
        strIn = ("What is the typical noise size you want to remove"
                 " (leave empty for current = "+str(lnoise0)+")?")
        lnoise0 = (raw_input(strIn) or lnoise0)
        strIn = ("What is the typical object size"
                 " (leave empty for current = "+str(lobject0)+")?")
        lobject0 = (raw_input(strIn) or lobject0)
        strIn = ("What is the size of the threshold box"
                 "(leave empty for current = "+str(boxSize0)+")?")
        boxSize0 = (raw_input(strIn) or boxSize0)
        print 'Please examine the processed image,  close it when done'
        img = cv.imread(fPath+tifList[num], -1)
        print fPath + tifList[num]
        img = cv.transpose(img)
        bwImg = processImage(img, scaleFact=1, sBlur=0.5, sAmount=0,
                             lnoise=lnoise0, lobject=lobject0,
                             boxSize=np.double(boxSize0), solidThres=0.65)
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(img)
        ax1.set_title('Raw Image')

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.imshow(bpass(img, lnoise0, lobject0))
        strIn = ("Filtered image. Noise size = " + str(lnoise0) + ". "
                 "Object size = "+str(lobject0)+". ")
        ax2.set_title(strIn)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.imshow(bwImg)
        ax3.set_title('Thresholded image. BoxSize value = '+str(boxSize0)+'. ')

        plt.show()
        satisfiedYN = raw_input(("Are you satisfied with the "
                                 "parameters? (yes or no) "))
        if satisfiedYN == 'yes':
            stop = True

    return lnoise0, lobject0, boxSize0


def splitRegions(fPath, numberOfRegions):
    """
    Performs a kmeans analysis to identify each regions.
    """
    import random as rnd
    import scipy.cluster.vq as vq
    lnoise0 = 1
    lobject0 = 8
    boxSize0 = 15
    tifList = []
    fileList = sort_nicely(os.listdir(fPath))
    for file in fileList:
        if file.endswith('tif'):
            tifList.append(file)
    num = rnd.randint(1, int(len(tifList)/10.))
    print num
    img = cv.imread(fPath+tifList[num], -1)
    img = cv.transpose(img)
    bwImg = processImage(img, scaleFact=1, sBlur=0, sAmount=0,
                         lnoise=lnoise0, lobject=lobject0,
                         boxSize=np.double(boxSize0), solidThres=0.65)
    posList = regionprops(bwImg)[:, 0:2]
    wPosList = vq.whiten(posList)

    idList = vq.kmeans2(wPosList[:, 1], numberOfRegions)
    x = np.zeros((numberOfRegions, ))
    for id in range(numberOfRegions):
        x[id] = np.mean(posList[idList[1] == id, 1])

    x = np.sort(x)
    xLimits = np.array([x[0]/2, (x[0]+x[1])/2])
    for id in range(1, numberOfRegions-1, 1):
        xLimits = np.vstack((xLimits, np.array([(x[id-1]+x[id])/2,
                            (x[id]+x[id+1])/2])))

    xLimits = np.vstack((xLimits, np.array([(x[id]+x[id+1])/2,
                        (x[-1]+img.shape[0])/2])))

    return xLimits.astype(np.int)


def saveListOfArrays(fname, listData):
    '''
    Save a list of different length arrays (but same width) as a single array
    '''
    arrayOut = np.hstack((0*np.ones((len(listData[0]), 1)), listData[0], ))
    for i in range(1, len(listData)):
        arrayTemp = np.hstack((i*np.ones((len(listData[i]), 1)),
                               listData[i],))
        arrayOut = np.vstack((arrayOut, arrayTemp))

    np.savez(fname, arrayOut)


def loadListOfArrays(fname):
    '''
    Loads an array generated by saveListOfArrays and export
    it as a list of arrays
    '''
    arrayInObj = np.load(fname)
    arrayIn = arrayInObj['arr_0']
    listLength = np.max(arrayIn[:, 0]) + 1

    listOut = []

    for i in range(int(listLength)):
        listOut.append(arrayIn[arrayIn[:, 0] == i, 1:])

    return listOut

#When run directly inside a folder containing images of cells,  this script
#creates the rawTrackData file.
if __name__ == "__main__":
    '''

    Options:
    -i,  --interactive (default): ask for the various prompts
    -n,  --no-interaction: processes the images and does the
        track analysis in the current directory
    '''

    INTERACTIVE = True

    for var in sys.argv:
        if (var == '-i'):
            INTERACTIVE = True
        elif (var == '--interactive'):
            INTERACTIVE = True
        elif (var == '-n'):
            INTERACTIVE = False
        elif (var == '--no-interaction'):
            INTERACTIVE = False

    lnoise, lobject, boxSize = 2, 8, 25

    if INTERACTIVE:
        FILEPATH = (raw_input(("Please enter the folder you want to analyze"
                    " (leave empty to use current location) ")) or "./")
        STABILIZEIMAGES = (raw_input(("Do you want to stabilize the images?"
                                     " (yes or no) ")))
        PROCESSFILES = raw_input(("Do you want to process the image files?"
                                  " (yes or no) "))
        if PROCESSFILES == 'yes':
            OPTIMIZEPROCESSING = raw_input(("Do you want to optimize the"
                                           " image processing parameters?"
                                           " (yes or no) "))
            if OPTIMIZEPROCESSING == 'yes':
                lnoise, lobject, boxSize = optimizeParameters(FILEPATH, 0)
        MULTIPLEREGIONS = raw_input(("Are there multiple disjointed regions"
                                     " in each image? (yes or no) "))
        if MULTIPLEREGIONS == 'yes':
            NUMBEROFREGIONS = int(raw_input(("How many regions per field"
                                             " of view? (Enter a number) ")))
        else:
            NUMBEROFREGIONS = 1
        LIMITFILES = raw_input(("Enter the number of files to analyze"
                                " (leave empty for all) ") or '0')
        if not LIMITFILES:
            LIMITFILES = 0
        print LIMITFILES
        LINKTRACKS = raw_input(("Do you want to link the cell tracks?"
                                " (yes or no) "))
        if LINKTRACKS == 'yes':
            PROCESSTRACKS = raw_input(("Do you want to analyze the cell"
                                       " tracks? (yes or no) "))
        SAVEPATH = (raw_input(("Please enter the location where the"
                               " analyzed files will be saved"
                               " (leave empty to use current location) "))
                    or "./")
    else:
        FILEPATH = './'
        STABILIZEIMAGES = 'no'
        PROCESSFILES = 'no'
        LINKTRACKS = 'yes'
        PROCESSTRACKS = 'yes'
        SAVEPATH = './'
        MULTIPLEREGIONS = 'no'
        NUMBEROFREGIONS = 1
        LIMITFILES = 0
        MATCHF = True

    np.savez(SAVEPATH+'processFiles.npz', lnoise=lnoise,
             lobject=lobject, boxSize=boxSize)

    if MULTIPLEREGIONS == 'yes':
        lims = splitRegions(FILEPATH, NUMBEROFREGIONS)
    else:
        lims = np.array([[0, -2]])
    if STABILIZEIMAGES == 'yes':
        translationList = stabilizeImages(FILEPATH, ew='tif', SAVE=False)
    else:
        translationList = 0
    if PROCESSFILES == 'yes':
        masterL, LL, AA = trackCells(FILEPATH, np.double(lnoise),
                                     np.double(lobject), np.double(boxSize),
                                     lims, int(LIMITFILES), translationList)
        for id in range(NUMBEROFREGIONS):
            saveListOfArrays(SAVEPATH+'masterL_'+str(id)+'.npz', masterL[id])
            saveListOfArrays(SAVEPATH+'LL_'+str(id)+'.npz', LL[id])

    if LINKTRACKS == 'yes':
        for id in range(NUMBEROFREGIONS):
            #masterL = loadListOfArrays(SAVEPATH+'masterL_'+str(id)+'.npz')
            #LL = loadListOfArrays(SAVEPATH+'LL_'+str(id)+'.npz')
            masterL = loadListOfArrays(SAVEPATH+'masterL_0.npz')
            LL = loadListOfArrays(SAVEPATH+'LL_0.npz')
            tr = linkTracks(masterL, LL)
            if PROCESSTRACKS == 'yes':
                tr = processTracks(tr, match=MATCHF)
            with open(SAVEPATH+'/trData_'+str(id)+'.dat',  'wb') as f:
                f.write(("# xPos yPos time cellID PoleID cellLength"
                         " cellWidth cellAngle avgIntensity"
                         " divisionEventsLeft divisionEventsRight"
                         " familyID cellAge OldestPoleAge"
                         " elongationRate\n"))
                np.savetxt(f, tr)
            print ("The analysis of region '+str(id)+' is complete."
                   " Data saved as "+SAVEPATH+"trData_"+str(id)+".dat")
