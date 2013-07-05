#!/usr/bin/python
'''
E. coli bacteria tracking suite
----------------------------------
Author: Guillaume Lambert

'''

import numpy as np
import re
import cv2 as cv
import matplotlib as mp
import matplotlib.nxutils as nx
import matplotlib.path as pa
import matplotlib.pylab as plt
import os as os
import munkres as mk
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
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

def unique_rows(a):
	""" Find the unique rows in an array """
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def smooth(x,windowLength=3):
	""" A simple moving average smoothing function """
	x=np.double(x)
	s=np.r_[2*x[0]-x[windowLength:1:-1],x,2*x[-1]-x[-1:-windowLength:-1]]
	w=np.ones(windowLength,'d')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[((windowLength-1)/2):-((windowLength-1)/2)]

def smoothCS(cs,windowLength):
	""" This functions smooths the result of cv.findContours """
	cs2=[]
	dd=[]
	for id in range(len(cs)):
		c=cs[id]
		cc=np.zeros((len(c),2))
		for id2 in range(len(c)):
			cc[id2,0]=c[id2][0][0]
			cc[id2,1]=c[id2][0][1]
		cc[:,0]=smooth(cc[:,0],windowLength)
		cc[:,1]=smooth(cc[:,1],windowLength)
		for id2 in range(len(c)):
			dd.append(np.array([[float(cc[id2,0]), 
				  float(cc[id2,1])]]))
		cs2.append(dd)
		cs2[id]=np.array(cs2[id])
	return cs2

def findAxes(m):
	""" Find the Minor and Major axis of an ellipse having moments m """
	Major=2*(2*(((m['mu20'] + m['mu02']) + 
		    ((m['mu20'] - m['mu02'])**2 + 
	              4*m['mu11']**2)**0.5))/m['m00'])**0.5
	Minor=2*(2*(((m['mu20'] + m['mu02']) - 
		    ((m['mu20'] - m['mu02'])**2 + 
		      4*m['mu11']**2)**0.5))/m['m00'])**0.5
	return Major,Minor

def anormalize(x):
	""" Normalize an array """
	y=x/np.sum(x)
	return y

def dot2(u, v):
	""" Returns the dot product of u and v """
	return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
        """|| u x v ||^2"""
	return sq2(u)*sq2(v) - dot2(u,v)**2

def sq2(u):
	""" Returns the magnitude of a vector """
        return dot2(u, u)

#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom)*db + b1

def resetIntersect(x0,y0,x1,y1):
	if (y0<0.):
		x0,y0=seg_intersect(array([double(x0),double(y0)]),
				    array([double(x1),double(y1)]),
				    array([-100000000.,0.]),
				    array([100000000.,0.]))
	if (y1<0.):
		x1,y1=seg_intersect(array([double(x0),double(y0)]),
				    array([double(x1),double(y1)]),
				    array([-100000000.,0.]),
				    array([100000000.,0.]))
	if (y0>1000.):
		x0,y0=seg_intersect(array([double(x0),double(y0)]),
				    array([double(x1),double(y1)]),
				    array([-100000000.,1000.]),
				    array([100000000.,10000.]))
	if (y1>1000.):
		x1,y1=seg_intersect(array([double(x0),double(y0)]),
				    array([double(x1),double(y1)]),
				    array([-100000000.,1000.]),
				    array([100000000.,10000.]))
	return x0,y0,x1,y1

def bpass(img,lnoise,lobject):
	'''
	imgOut = bpass(img,lnoise,lobject) return an 
	image filtered according to the lnoise (size of the noise) 
	and the lobject (typical size of the object in the image) parameter.
	
	The script has been translated from the bpass.m script developed 
	by John C. Crocker and David G. Grier.'''
	
	lnoise=np.double(lnoise)
	lobject=np.double(lobject)
	image_array=np.double(img)  #Convert the input image to double
	
	#Create the gaussian and boxcar kernels
	gauss_kernel_range=np.arange((-5*lnoise),(5*lnoise))/(2*lnoise)
	gauss_kernel=anormalize(np.exp(-np.power(gauss_kernel_range,2)))
	boxcar_kernel=anormalize(np.ones((1,len(np.arange(-lobject,lobject))),
		                         dtype=float))
	#Apply the filter to the input image
	gconv0=cv.filter2D(np.transpose(image_array),-1,gauss_kernel)
	gconv=cv.filter2D(np.transpose(gconv0),-1,gauss_kernel)
	bconv0=cv.filter2D(np.transpose(image_array),-1,boxcar_kernel)
	bconv=cv.filter2D(np.transpose(bconv0),-1,boxcar_kernel)
	#Create the filtered image
	filtered=gconv-bconv
	#Remove the values lower than zero
	filtered[filtered<0]=0
	#Output the final image
	imgOut=filtered.copy()
	return imgOut

def unsharp(img,sigma=5,amount=10):
	'''
	imgOut = unshapr(img,sigma=5,amount=10) create an unsharp 
	operation on an image. 
	If amount=0, create a simple gaussian blur with size sigma.
	'''
	if sigma:
		img=np.double(img)
		imgB=cv.GaussianBlur(img,(img.shape[0]-1,img.shape[1]-1),sigma)
		imgS = img*(1+amount) + imgB*(-amount)
		if amount:
			return imgS
		else:
			return imgB
	else:
		return img
	
def regionprops(bwImage,scaleFact=1):
	'''Replicate MATLAB's regionprops script.	
	STATS = regionprops(bwImage,scaleFact=1) returns the following 
	properties in the STATS array:
	STATS=[xPosition, yPosition, MajorAxis, 
	       MinorAxis, Orientation, Area, Solidity]	 
	'''
	#import the relevant modules
	#Find the contours
	bwI=np.uint8(bwImage.copy())
	csr,_ = cv.findContours(bwI.copy(),
				mode=cv.RETR_TREE, 
				method=cv.CHAIN_APPROX_SIMPLE)
	numC=int(len(csr))
	#Initialize the variables
	k=0
        for i in range(numC):
                if len(csr[i])>=5:
			k=k+1
	majorAxis=np.zeros((k,1),dtype=float)
	minorAxis=np.zeros((k,1),dtype=float)
	Orientation=np.zeros((k,1),dtype=float)
	EllipseCentreX=np.zeros((k,1),dtype=float)
	EllipseCentreY=np.zeros((k,1),dtype=float)
	Area=np.zeros((k,1),dtype=float)
	Solidity=np.zeros((k,1),dtype=float)
	k=0
	for i in range(numC):
		if len(csr[i])>=5:
			#Orientation
			centroid,axes,angle = cv.fitEllipse(csr[i])
			Orientation[k]     = angle
			#Major,Minor axes
			#Centroid
			m = cv.moments(csr[i])
			if m['m00']>0:			
				centroid      = (m['m10']/m['m00'], 
						 m['m01']/m['m00'])
				majorAxis[k],minorAxis[k]=findAxes(m)
				EllipseCentreX[k]  = centroid[0] # x,y
				EllipseCentreY[k]  = centroid[1]
				#Solidity&Area
				Area[k] = m['m00']
				# CONVEX HULL stuff
				# convex hull vertices
				ConvexHull    = cv.convexHull(csr[i])
				ConvexArea    = cv.contourArea(ConvexHull)
				# Solidity := Area/ConvexArea
				Solidity[k]   = np.divide(Area[k],ConvexArea)		
			k=k+1
			
	allData=np.zeros((k,7))
	allData[:,0]=EllipseCentreX.squeeze()/scaleFact
	allData[:,1]=EllipseCentreY.squeeze()/scaleFact
	allData[:,2]=majorAxis.squeeze()/scaleFact
	allData[:,3]=minorAxis.squeeze()/scaleFact
	allData[:,4]=Orientation.squeeze() 
	allData[:,5]=Area.squeeze()/scaleFact**2 
	allData[:,6]=Solidity.squeeze() 
	return np.double(allData)

def bwlabel(bwImg):
	'''Replicate MATLAB's bwlabel function.
	labelledImg = bwlabel(bwImg) takes a black&white image and 
	returns an image where each connected region is labelled 
	by a unique number.
	'''

	#Import relevant modules
	#Change the type of image
	bwImg2=np.uint8(bwImg.copy())
	bw=np.zeros(bwImg2.shape)
	#Find the contours, count home many there are
	csl,_ = cv.findContours(bwImg2.copy(),
				mode=cv.RETR_TREE, 
				method=cv.CHAIN_APPROX_SIMPLE)
	numC=int(len(csl))
	#Label each cell in the figure
	k=0
	for i in range(numC):
		if len(csl[i])>=5:
			k=k+1
			cv.drawContours( bw, csl, i, k, thickness=-1)
		else:
			cv.drawContours( bw, csl, i, 0, thickness=-1)
	return np.uint16(bw)

def traceOutlines(trT,dist,size):
	'''
	ImgOut=traceOutlines(trT) Trace the outline of every cell 
	in the array trT (trT should contain the tr data of a single time).
	'''

	bw=np.zeros(size)
	csl=[]
	for pt in trT:
		boundPt=getBoundingBox(pt,dist)
		dp=np.array(([[[int(round(boundPt[0,0])),int(round(boundPt[0,1]))]]]),dtype=np.int32)
		for p in boundPt:
			dp=np.vstack((dp,np.array(([[[int(round(p[0])),int(round(p[1]))]]]),dtype=np.int32)))
		csl.append(dp)
	k=0
	numC=int(len(csl))
	for i in range(numC):
		if len(csl[i])>0:
			k=k+1	
			cv.drawContours(bw,csl,i,k,thickness=-1)
		else:
			cv.drawContours(bw,csl,i,0,thichness=-1)
	return bw

def removeSmallBlobs(bwImg,bSize=10):
	'''
	imgOut=removeSmallBlobs(bwImg,bSize) removes processes a binary image
	and removes the blobs which are smaller than bSize (area).
	'''
	bwImg2=np.uint8(bwImg.copy())
	bw=np.zeros(bwImg2.shape)

        #Find the contours, count home many there are
        csl,_ = cv.findContours(bwImg2.copy(),
                                mode=cv.RETR_TREE,
                                method=cv.CHAIN_APPROX_SIMPLE)
        numC=int(len(csl))
	#Label each cell in the figure which are smaller than bSize
        for i in range(numC):
		area = cv.contourArea( csl[i]);
                if area<=bSize:
                        cv.drawContours( bw, csl, i, 1, thickness=-1)
                else:
                        cv.drawContours( bw, csl, i, 0, thickness=-1)
	maskBW=1-bw 
	return np.uint8(bwImg2*maskBW)

def floodFill(imgIn,seedPt,pixelValue):
	'''
	This script perform a flood fill, starting from seedPt.
	'''
	labelledImg=bwlabel(imgIn)

	regionID=labelledImg[seedPt[0],seedPt[1]]

	bwImg0=imgIn.copy()
	bwImg0[labelledImg==regionID]=pixelValue		

	return bwImg0

def drawConvexHull(bwImg):
	        #Import relevant modules
        #Change the type of image
        bwImg2=np.uint8(bwImg.copy())
        bw=np.zeros(bwImg2.shape)
        #Find the contours, count home many there are
        csl,_ = cv.findContours(bwImg2.copy(),
                                mode=cv.RETR_TREE,
                                method=cv.CHAIN_APPROX_SIMPLE)
        numC=int(len(csl))
        #Label each cell in the figure
        k=0
        for i in range(numC):
                if len(csl[i])>=5:
			chl=cv.convexHull(csl[i])
                        k=k+1
                        cv.drawContours( bw, [chl], 0, 1, thickness=-1)
                else:
                        cv.drawContours( bw, csl, i, 0, thickness=-1)
        return np.uint16(bw)


def avgCellInt(rawImg,bwImg):
	'''
	STATS = avgCellInt(rawImg,bwImg) return an array containing 
	the pixel value in rawImg of each simply connected region in bwImg.
	'''
	bwImg0=np.uint8(bwImg.copy())
	bw=np.zeros(bwImg0.shape)
	
	csa,_ = cv.findContours(bwImg0, 
				mode=cv.RETR_TREE, 
				method=cv.CHAIN_APPROX_SIMPLE)
	numC=int(len(csa))
	k=0
	for i in range(0,numC):
		if len(csa[i])>=5:
			k=k+1
	avgCellI=np.zeros((k+1,1),dtype=float)
	
	k=0
	for i in range(0,numC):
		if len(csa[i])>=5:
			# Average Pixel value		
			bw=np.zeros(bwImg0.shape)
			k=k+1
			cv.drawContours( bw, csa, i, 1, thickness=-1)
			regionMask = (bw==(1))
			avgCellI[k]=np.sum(rawImg*regionMask)/np.sum(regionMask)
	return np.double(avgCellI)


def segmentCells(bwImg,propIndex,pThreshold,iterN=2):
	'''
	imgOut = segmentCells(bwImg,propIndex,pThreshold,iterN=2) applies 
	a watershed transformation to the regions in bwImg whose property 
	propIndex are smaller than pThreshold.
	'''
	labelImg=bwlabel(bwImg).copy()
	lowImg=np.double(np.zeros((np.size(labelImg,0),np.size(labelImg,1))))
	lowSIndex=np.nonzero(propIndex<pThreshold)
	if lowSIndex[0].any():
		for id in np.transpose(lowSIndex):
			lowImg = lowImg + np.double(labelImg==(id+1))
		lowImg=np.uint16(lowImg)
		markers=cv.dilate(lowImg,None,iterations=iterN) + \
			cv.erode(lowImg,None,iterations=iterN)
		markers32=np.int32(markers)
		rgbImg=np.zeros((np.size(lowImg,0),np.size(lowImg,1),3))
		rgbImg[:,:,0]=np.uint8(lowImg>0)
		rgbImg[:,:,1]=np.uint8(lowImg>0)
		rgbImg[:,:,2]=np.uint8(lowImg>0)	
		cv.watershed(np.uint8(rgbImg),markers32)
		m=cv.convertScaleAbs(markers32)-1
#		m=cv.dilate(m,None,iterations=1)
		m=dilateConnected(m,iterN)
		maskImg=(np.ones(((np.size(lowImg,0),np.size(lowImg,1)))) - 
			 m + lowImg)==1
		segImg=np.multiply(maskImg,labelImg)>0
	else:
		return bwImg
	return np.double(segImg)

def dilateConnected(imgIn,nIter):
	"""
	imgOut = dilateConnected(imgIn,nIter) dilates a binary image 
	while preserving the number of simply connected domains.
	
	nIter is the dilation factor 
	(number of times the dilate function is applied)
	"""
	bwImgD=np.uint8(imgIn.copy())
	imgOut=np.double(imgIn*0)
	bwLD=bwlabel(bwImgD)
	for i in range(1,bwLD.max()+1):
		imgOut=imgOut +  np.double(cv.dilate(np.uint16(bwLD==i),
			                   None,iterations=(nIter+2)))
	dilImg=cv.dilate(bwImgD,None,iterations=nIter)
	skelBnd = skeletonTransform(np.uint8(imgOut>1))
	skelBnd = cv.dilate(skelBnd,None,iterations=1)

	imgOut=np.double(dilImg) - skelBnd*(bwLD==0)
	imgOut=imgOut>0
	return np.double(imgOut)	

def skeletonTransform(bwImg):
	'''
	Generate the skeleton transform of a binary image
	Based on: 
	http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
	'''
	element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
	done = False
	img = np.uint8(bwImg.copy())
	skel = img*0
	while not done:
		eroded = cv.erode(img,element)
		temp = cv.dilate(eroded,element)
		temp = img - temp
		skel = cv.bitwise_or(skel,temp)	
		img = eroded.copy()
		
		if not cv.countNonZero(img):
			done = True
	#Remove isolated pixels

	skel = skel - cv.filter2D(skel,-1,np.array([[-9,-9,-9],[-9,1,-9],[-9,-9,-9]],dtype=float))

	return skel

def drawVoronoi(points,imgIn):
	"""
	inspired by http://stackoverflow.com/questions/10650645/
	python-calculate-voronoi-tesselation-from-scipys-delaunay-triangulation-in-3d
	"""
	from scipy.spatial import Delaunay
	# Draw a BW image of a voronoi tesselation,
	# with the lines having a value of 1 and the rest 0

	imgIn=np.double(imgIn)
	
	lenX=double(size(imgIn,1))
	lenY=double(size(imgIn,0))

	tri=Delaunay(points)
	# Triangle vertices
	
	p = tri.points[tri.vertices]

	A = p[:,0,:].T
	B = p[:,1,:].T
	C = p[:,2,:].T
	a = A - C
	b = B - C

	cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C

	# Grab the Voronoi edges
	vc = cc[:,tri.neighbors]
	vc[:,tri.neighbors == -1] = np.nan # edges at infinity, 
					   # plotting those would need more work...

	lines = []
	lines.extend(zip(cc.T, vc[:,:,0].T))
	lines.extend(zip(cc.T, vc[:,:,1].T))
	lines.extend(zip(cc.T, vc[:,:,2].T))

	

	for lin in lines:
		if not np.sum(np.isnan(lin)):
			x0=(lin[0][0])
			y0=(lin[0][1])
			x1=(lin[1][0])
			y1=(lin[1][1])
			x0,y0,x1,y1=resetIntersect(x0,y0,x1,y1)
			cv.line(imgIn,(int(x0),int(y0)),(int(x1),
				int(y1)),(1,1,1),thickness=2)
	return imgIn


def labelOverlap(img1,img2):
	'''
	areaList = labelOverlap(img1,img2) finds the overlapping 
	cells between two images. 
	
	Returns areaList as [label1 label2 area]
	img1 and img2 should be generated with bwlabel.
	'''
	#Find the overlapping indices	
	overlapImg=np.uint8((img1>0)*(img2>0))
	index1=overlapImg*np.uint16(img1)
	index2=overlapImg*np.uint16(img2)

	#Store the index couples in a list
	indexList=np.vstack([index1.flatten(0),index2.flatten(0)])
	
	#remove the [0,0] rows, convert to 1D structured array
	indexList=np.uint16(np.transpose(indexList[:,indexList[0,:]>0]))
	indexList=indexList.view(','.join(2 * ['i2']))
	unique_vals, indices = np.unique(indexList,return_inverse=True)
	if len(indices)>0:
		counts = np.bincount(indices)
		numId=len(counts)
		areaList=np.zeros((numId,3))
		for id in range(numId):
			areaList[id,0]=unique_vals[id][0]
			areaList[id,1]=unique_vals[id][1]
			areaList[id,2]=counts[id]
	else:
		areaList=np.zeros((1,3))
	return areaList

def matchIndices(dataList,matchType='Area'):
	'''
	linkList=matchIndices(dataList,matchType='Area') uses the 
	Munkres algorithm to performs an assignment problem from the 
	data in dataList=[id1,id2,distanceValue]. 
	
	*if matchType='Area', it can be used to match the cell areas 
	between successive frames (favorizes largest area value)
	
	*if matchType='Distance', it can be used to match the cell 
	positions between successive frames (favorises smallest distance value)
	'''
	linkList=[0,0,0];
	#Loop over each indices to find its related matches.
	for cellID in np.transpose(np.unique(dataList[:,0])):
		stop=0
		#check is dataList is empty
		if np.size(dataList,0)>0:
			matchData=dataList[dataList[:,0]==cellID,:]
		else:
			matchData=[]
		if np.size(matchData,0)>0:
			#Collect related cells
			while stop==0:
				matchData0=np.array(matchData)
				for i in np.transpose(np.unique(matchData[:,1])):
					matchData=np.vstack([matchData,
						             dataList[dataList[:,1]==i,:]])
				for i in np.transpose(np.unique(matchData[:,0])):
					matchData=np.vstack([matchData,
						             dataList[dataList[:,0]==i,:]])
				matchData=unique_rows(matchData)
				if np.array_equal(matchData,matchData0):
					stop=1
			#Create a label for each cell 
			# (instead of the cell ID, which may have gaps)	
			k0=0
			assign0=np.zeros((np.max(matchData[:,0])+1))
			assign1=np.zeros((np.max(matchData[:,1])+1))
			for i in np.transpose(np.unique(matchData[:,0])):
				k0=k0+1
				assign0[i]=k0
			k1=0
			for i in np.transpose(np.unique(matchData[:,1])):
				k1=k1+1
				assign1[i]=k1	
			
			k=np.max([k0,k1])
			#Create the linear assignment matrix
			A=np.zeros((k,k))
			for i in range(len(matchData[:,0])):
				if matchType is 'Area':
					indx1=assign0[matchData[i,0]]-1
					indx2=assign1[matchData[i,1]]-1
					A[indx1][indx2]=matchData[i,2]	
				elif matchType is 'Distance':
					if matchData[i,2]>0:
						indx1=assign0[matchData[i,0]]-1
						indx2=assign1[matchData[i,1]]-1
						A[indx1][indx2]=1./matchData[i,2]
			#Find matching elements with munkres algorithm
			m=mk.Munkres()
			#inds=m.compute(-A)
			inds,inds0=hun.lap(-A)
			inds2=np.zeros((len(inds),2))
			for kk in range(len(inds)):
				inds2[kk,0]=kk
				inds2[kk,1]=inds[kk]		
			inds=inds2.copy()
		
			linkList0=np.zeros((len(matchData[:,0]),3))
			#Put result into the linkList0 array
			for i in range(len(inds)):
				if ((inds[i][0]+1)<=k0)&((inds[i][1]+1)<=k1):
					linkList0[i,0]=np.nonzero(assign0==(inds[i][0]+1))[0][0]
					linkList0[i,1]=np.nonzero(assign1==(inds[i][1]+1))[0][0]
					if matchType is 'Area':
						linkList0[i,2]=A[inds[i][0]][inds[i][1]]
					elif matchType is 'Distance':
						if A[inds[i][0]][inds[i][1]]>0:
							linkList0[i,2]=1/A[inds[i][0]][inds[i][1]]

			linkList0=linkList0[linkList0[:,2]>0,:]
			#Append data at the end of linkList
			linkList=np.vstack([linkList,linkList0])
			#Remove the matched data from the input list
			for id in np.transpose(linkList0[:,0]):
				if np.size(dataList[:,0]!=id,0)>0:
					dataList=dataList[dataList[:,0]!=id,:]
			for id in np.transpose(linkList0[:,1]):
				if np.size(dataList[:,1]!=id,0)>0:
					dataList=dataList[dataList[:,1]!=id,:]
	linkList=linkList[linkList[:,0]>0,:]
	return linkList

def putLabelOnImg(fPath,tr,dataRange,dim,num):
	'''
	putLabelOnImg(fPath,tr,dataRange,dim) goes through the images in 
	folder fPath over the time in dataRange and adds on top of each 
	cell the values stored along dimension=dim of tr.
	'''
	#Initialize variables
	tifList=[] 
        fileNum0=0      
        fileNum=0
	k=0
	trTime=splitIntoList(tr,2)
        #Import and process image file
        fileList=sort_nicely(os.listdir(fPath))
        for file in fileList:
                if file.endswith('tif'):
                        tifList.append(file)
	if not len(dataRange):
		dataRange=range(len(tifList))
	for t in dataRange:
		time_start=time.time()
		if tifList:
			fname = tifList[t] 
			print fPath+fname
	                img=cv.imread(fPath+fname,-1)
	                img=cv.transpose(img)
			bwImg=processImage(img,scaleFact=1,sBlur=0.5,
				   sAmount=0,lnoise=1,lobject=8,
				   thres=2,solidThres=0.65,
				   lengthThres=1.5,widthThres=20)
			plt.imshow(bwImg)
			bwI=np.double(bwImg.copy())
		plt.hold(True)
		trT=trTime[t]
		if np.size(trT)>1:
			img=traceOutlines(trT,0.85,(250,1200))
			plt.imshow(img)	
			for cell in range(len(trT[:,3])):
				'''plt.text(trT[cell,0],trT[cell,1],
				            str(trT[cell,dim])+', '+str(trT[cell,3]),
					    color='w',fontsize=6)
				'''
				#boxPts=getBoundingBox(trT[cell,:])
				#plt.plot(boxPts[:,0],boxPts[:,1],'gray',lw=0.5)
				
				plt.text(trT[cell,0],trT[cell,1],
					 str(trT[cell,dim]),color='k',fontsize=5)
				for c in num:					
					if trT[cell,dim]==c:
						plt.text(trT[cell,0],trT[cell,1],
						 str(trT[cell,dim]),color='r',fontsize=5)

#				if bwI[np.floor(trT[cell,1]),np.floor(trT[cell,0])]>0:
#					bwI=floodFill(bwI.copy(),(trT[cell,1],trT[cell,0]),trT[cell,9])
			
#		plt.imshow(bwI)
#		plt.clim((0,30))
		plt.title(str(t))
		plt.hold(False)
		plt.xlim((0,950))
		plt.ylim((25,220))
		
#		cv.imwrite(fPath+'Fig'+str(t)+'.jpg',np.uint8(bwI))
#		plt.savefig(fPath+"Fig"+str(t)+".png",dpi=(120))
		plt.show()
		plt.draw()
		plt.clf()	
	
def processImage(imgIn,scaleFact=1,sBlur=0.5,sAmount=0,lnoise=1,
		 lobject=8,thres=3,solidThres=0.65,
		 lengthThres=1.5,widthThres=20):
	'''
	This is the core of the image analysis part of the tracking algorithm.
	bwImg=processImage(imgIn,scaleFact=1,sBlur=0.5,sAmount=0,lnoise=1,
	                   lobject=8,thres=3,solidThres=0.65,lengthThres=1.5,
			   widthThres=20) 
	returns a black and white image processed from 
		a grayscale input (imgIn(.
	scaleFact: scales the image by this factor prior 
		to performing the analysis
	sBlur = size of the characteristic noise you want 
		to remove with a gaussian blur.
	sAmount = magnitude of the unsharp mask used to process the image
	lnoise = size of the typical noise when segmenting the image
	lobject = typical size of the pbject to be considered
	thres = intensity threshold used to create the black and white image
	solidThres = size of the lowest allowed cell solidity
	lengthThres = length of the cell at which you expect cell 
		to start dividing (1.5 = 1.5 times the average length)
	widthThres = width of the cells. Cells above that width will be 
		segmented as they may represent joined cells
	'''	
	img=cv.resize(imgIn,(np.size(imgIn,1)*scaleFact,
		      np.size(imgIn,0)*scaleFact))
	imgU=(unsharp(img,sBlur,sAmount))
	imgB=bpass(imgU,lnoise,lobject).copy()
	bwImg=np.uint8(np.double(imgB)>thres)
	#Dilate cells while maintaining them unconnected
	#Segment Cells accorging to solidity
	bwImg=segmentCells(bwImg>0,regionprops(bwImg>0,scaleFact)[:,6],
			   solidThres,2)	
	#Segment cells according to length
	mLength=np.mean(regionprops(bwImg>0,scaleFact)[:,2])
	bwImg=segmentCells(bwImg>0,-regionprops(bwImg>0,scaleFact)[:,2],
			   -lengthThres*mLength,2)	
	bwImg=removeSmallBlobs(bwImg,100)
	bwImg=dilateConnected(bwImg,2)
	return bwImg
	#Segment cells according to width (currently not used)
	bwImg=segmentCells(bwImg>0,-regionprops(bwImg>0,scaleFact)[:,3],
			   -widthThres,3)	
	bwImg=np.uint8((bwImg>0))
	
	return bwImg

def preProcessCyano(brightImg,chlorophyllImg):
	'''
	Pre-process the Chlorophyll and brightfield images so that they can be
	segmented directly.
	''' 
	solidThres=0.75
	cellMask = cv.dilate(np.uint8(bpass(chlorophyllImg,1,10)>50),None,iterations=15)
	processedBrightfield = bpass(brightImg,1,10)>25

	dilatedIm=cv.dilate(removeSmallBlobs(processedBrightfield*cellMask,75),None,iterations=2)
#	dilatedIm=(removeSmallBlobs(processedBrightfield*cellMask,15))
	if np.sum(cellMask==0):
		seedPt = ((1-cellMask).nonzero()[0][10],(1-cellMask).nonzero()[1][10])

		imgOut=dilateConnected(1-floodFill(dilatedIm,seedPt,1),2)
#		imgOut=(1-floodFill(dilatedIm,seedPt,1))
	else:
		imgOut=dilateConnected(1-dilatedIm,2)

	imgOut=np.uint8(processImage(imgOut.copy(),thres=0))

#	imgOut=processImage(-imgOut+(bpass(chlorophyllImg,1,10)>10),thres=0)

#	imgOut=removeSmallBlobs(imgOut,75)	
	#Segment Cells accorging to solidity
#	imgOut=segmentCells(imgOut>0,regionprops(imgOut>0,1)[:,6],
#			   solidThres,2)	
	return np.uint8(imgOut)

def trackCells(fPath,lnoise=1,lobject=8,thres=3,lims=np.array([[0,-1]]),maxFiles=0):
	'''
	This prepares the data necessary to track the cells.
	masterL,LL,AA=trackCells(fPath,lnoise=1,lobject=8,thres=5,lims=0)
	processes the files in fPath.
	
	lims is a Nx2 list containing the limits of each subregions.
	'''
	#Initialize variables
	sBlur=0
	sAmount=0
	bPassNum=1
	scaleFact=1
	solidThres=0.65
	lengthThres=1.5
	tifList=[]
	masterList=[]
	LL=[];
	AA=[]
	fileNum0=0
	fileNum=0
	#Import and process image file
	fileList=sort_nicely(os.listdir(fPath))
	for file in fileList:
		if file.endswith('tif'):
			tifList.append(file)
	k=0
	if maxFiles>0:
		tifList=tifList[:maxFiles]
	kmax=len(tifList)
	startProgress('Analyzing files:')
	regionP=list(range(len(lims)))
	areaList=list(range(len(lims)))
	linkList=list(range(len(lims)))
	masterList=list(range(len(lims)))
	AA=list(range(len(lims)))
	LL=list(range(len(lims)))
	imgCropped=list(range(len(lims)))
	bwImg=list(range(len(lims)))
	bwL=list(range(len(lims)))
	bwL0=list(range(len(lims)))
	for fname in np.transpose(tifList):
		print(fPath+fname)
		k=k+1
		progress(np.double(k)/np.double(kmax))
		img0=cv.imread(fPath+fname,-1)
		if len(img0)==0:
			img=img.copy()
		else:
			img=img0.copy()	
		img=cv.transpose(img)
		for id in range(len(lims)):
			imgCropped[id]=img[lims[id,0]:lims[id,1],:]
			bwImg[id]=processImage(imgCropped[id],scaleFact,sBlur,
					   sAmount,lnoise,lobject,
					   thres,solidThres,lengthThres)
			bwL[id]=bwlabel(bwImg[id])
			fileNum=int(re.findall(r'\d+',fname)[0])
			if bwL[id].max()>5:
				
				regionP[id]=regionprops(bwImg[id],scaleFact)
				avgCellI=avgCellInt(imgCropped[id].copy(),bwImg[id].copy())
				if np.isnan(avgCellI).any():
					avgCellI[np.isnan(avgCellI)]=0
				regionP[id]=np.hstack([regionP[id],avgCellI[1:]])
				if (fileNum-fileNum0)==1:
					areaList[id]=labelOverlap(bwL0[id],bwL[id])
					AA[id].append(areaList[id])			
					linkList[id]=matchIndices(areaList[id],'Area')
					LL[id].append(linkList[id])		
				#Extract regionprops
				if fileNum0==0:
					masterList[id]=[regionP[id]]
					AA[id]=[]
					LL[id]=[]
				else:
					masterList[id].append(regionP[id])
				bwL0[id]=bwL[id].copy()

		fileNum0=fileNum+0
	endProgress()
	return masterList,LL,AA

def updateTR(M,L,time):
	'''
	trTemp=updateTR(M,L,time)
	This function orders the values in masterL (M) from L at t=time.
	'''
	trTemp=np.zeros((len(L[time]),8))
	parentIndex=(L[time][:,0]-1).tolist()
	trTemp[:,[0,1,3,4,5,6,7]]=M[time][parentIndex,:].take([0,1,8,2,3,4,7],
			                                      axis=1)
	return trTemp
	
def linkTracks(masterL,LL):
	'''
	tr=linkTracks(masterL,LL) links the frame-to-frame tracks.
	It returns an array tr=[xPos,yPos,time,cellId,length,width,
				orientation,pixelIntensity,divisionEvents,
				familyID,age]
	'''
	#Links the frame-to-frame tracks 
	
	totalTime=len(LL)
	ids=np.arange(len(LL[0]))+1
	masterL[0]=np.hstack((masterL[0],np.zeros((len(masterL[0]),1))))
	masterL[0][(LL[0][:,0]-1).tolist(),8]=ids
	tr=updateTR(masterL,LL,0)
	tr[:,2]=0
	maxID=np.max(ids)
	tr0=np.zeros((len(LL[1]),8))
	print "Linking cell overlaps"
	for t in range(totalTime-1):
		tr0=np.zeros((len(LL[t+1]),8))
		masterL[t+1]=np.hstack([masterL[t+1],
			                np.zeros((len(masterL[t+1]),1))])
		masterL[t+1][(LL[t][:,1]-1).tolist(),8]=ids
		tr0=updateTR(masterL,LL,t+1)
		tr0[:,2]=t+1
		ids=masterL[t+1][(LL[t+1][:,0]-1).tolist(),8]
		for i in range(len(ids)):
			if ids[i]==0:
				maxID=maxID+1
				ids[i]=maxID
				tr0[i,3]=maxID
		tr=np.vstack([tr,tr0])
	tr=tr[tr[:,2].argsort(-1,'mergesort'),]
	tr=tr[tr[:,3].argsort(-1,'mergesort'),]

	return tr

def processTracks(trIn):
	"""
	tr=processTracks(trIn) returns a track with linked mother/daughter
	labels (tr[:,8]), identified family id (tr[:,10]), cell age (tr[:,11])
	and the elongation rate (tr[:,12])
	"""
	#Find and assign the daughter ID at each division 
	# (do it twice because the first time around is reorganizes the 
	# trackIDs so that the mother is on the left).
	

	print "Smoothing Length"
	tr=smoothLength(trIn)
		
#	tr=removeShortTracks(tr,1)	
	print "Bridging track gaps"	
	tr=joinTracks(tr,2.5,[4,6])
	
	print "Splitting Tracks"
	tr=splitTracks(tr)
	print "Finding division events, Merging tracks"
	tr=mergeTracks(tr)	

#	print "rename Tracks"
	#Remove orphaned cells and cells that do not divide
#	tr=renameTracks(tr)	

	print "find family IDs"
	tr=findFamilyID(tr)
	print "match family IDs"
#	tr=matchFamilies(tr)	
	tr=matchFamilies(tr)	

	print "fix family IDs"
	#tr=fixFamilies(tr)	
	print "Adding cell Age"
	tr=addCellAge(tr)
	
	
	print "Compute elongation rate"	
	tr=addElongationRate(tr)


	return tr

def getBoundingBox(param,dist=1):
	'''
	boxPts=getBoundingBox(param,dist=1)
	Return the polygon that bounds an ellipse defined by 
	param = [xPos,yPos,time,id,length,orientation]
	The parameter dist denotes the factor by which the length 
	and width are multiplied by when creating the bounding box.
	'''	
	#Define parameters
	Pos=[param[0],param[1]]
	SemiMajor=dist*param[4]
	SemiMinor=dist*param[5]
	Orient=param[6]*np.pi/180
	
	RotMatrix=[[np.cos(Orient),-np.sin(Orient)], [np.sin(Orient),np.cos(Orient)]]
	boxPts=[[SemiMinor/2,SemiMajor/2],
		[-SemiMinor/2,SemiMajor/2],
		[-SemiMinor/2,-SemiMajor/2],
		[SemiMinor/2,-SemiMajor/2],
		[SemiMinor/2,SemiMajor/2]]
	for elem in range(len(boxPts)):
		boxPts[elem]=np.dot(RotMatrix,boxPts[elem])+[Pos[0],Pos[1]]
	
	return np.array(boxPts)

def eucledianDistances(xy1, xy2):
	'''
	dist=eucledianDistances(xy1,xy2) returns the eucledian distances 
	between the set of xy-points xy1 and xy2.
	xy should be n-by-2 arrays containing [xPos,yPos]
	'''
	d0 = np.subtract.outer(xy1[:,0], xy2[:,0])
	d1 = np.subtract.outer(xy1[:,1], xy2[:,1])
	return np.hypot(d0,d1)

def fixTracks(tr,dist,timeRange):
	'''
	listOfMatches=fixTracks(tr,dist,timeRange) finds the trackIDs 
	that join track ends with nearby track starts.
	listOfMatches return [id1, id2, time, distance]
	
	dist is the factor by which the box bounding a cells is 
	multiplied by when looking for possible candidates.

	timeRange is given by [t1,t2] and denotes how far in the 
	past (t1) and future (t2) should the script check to find 
	the possible candidates.
	'''
	#Find track ends	
	masterEndList=tr[np.diff(tr[:,3])>0,:]
	
	#Find track starts
	masterStartList=tr[np.roll(np.diff(tr[:,3])>0,1),:]
	
	listOfMatches=matchTracksOverlap(masterEndList,masterStartList,
				  dist,timeRange)

	return listOfMatches	
	#Reassign the track IDs

def joinTracks(trIn,dist,dataRange):
	'''
	Merge the track ends with track starts. Do not care about cell divisions.
	'''	

	mList0=fixTracks(trIn,dist,dataRange)

	matchData=matchIndices(mList0,'Area')


	trOut=mergeIndividualTracks(trIn.copy(),matchData.copy())

	trOut=smoothLength(trOut)
	
	return trOut

def reassignTrackID(tr,matchList):
	'''
	tr=reassignTrackID(tr,matchList) use the output of fixTracks 
	to reassing the proper id to matched tracks
	'''
	if len(matchList)<2:
		return tr
	shortTrack=3
	#Match the indices of the link candidates
	#matchL=matchIndices(matchList.take([0,1,3],axis=1),'Distance')
	matchL=matchList.copy()

	tr2=tr.copy()
	
	for id in range(len(matchL)):
		
		#Get indices
		id0=matchL[id,0]
		id1=matchL[id,1]
		
		#Correct IDs in track
		tr2[tr[:,3]==id1,3]=id0

		if tr2.shape[1]>=9:
			tr2[tr[:,8]==id1,8]=id0
			tr2[tr[:,8]==-id1,8]=-id0
			tr2[tr[:,9]==id1,9]=id0
			tr2[tr[:,9]==-id1,9]=-id0
	
		#Correct IDs in matchL
		matchL[matchL[:,0]==id1,0]=id0

        tr2=tr2[tr2[:,2].argsort(-1,'mergesort'),]
        tr2=tr2[tr2[:,3].argsort(-1,'mergesort'),]

	tr2[(np.diff(tr2[:,3])==0)&(np.diff(tr2[:,2])==0),:]=0

	trOut=tr2[tr2[:,0]>0,:]

	return tr2

def removeShortCells(tr,shortCells=6):
	'''
	tr=removeShortTracks(tr,shortTrack=3) removes 
	tracks shorter than shortTrack
	'''
	
	shortIDs=np.unique(tr[tr[:,4]<shortCells,3])

	trI=splitIntoList(tr,3)

	for i in shortIDs:
		trI[int(i)]=[]

	trS=revertListIntoArray(trI)

	return trS
	


def removeShortTracks(tr,shortTrack=3):
	'''
	tr=removeShortTracks(tr,shortTrack=3) removes 
	tracks shorter than shortTrack
	'''
	trLength=np.histogram(tr[:,3],np.unique(tr[:,3]))
        idShort=trLength[1][trLength[0]<=shortTrack]
        k=0
	trT=splitIntoList(tr,3)
        #Reassign cell ID
        for id in idShort:
                trT[int(id)]=[]
	
	trS=revertListIntoArray(trT)
	return trS
	
def matchTracksOverlap(inList1,inList2,dist,timeRange):
	matchList=np.zeros((1,3))


	for pts in inList1:
		boxPts=getBoundingBox(pts,dist)
		nearPts=inList2[(inList2[:,2]<=(pts[2]+timeRange[1])) &
				(inList2[:,2]>=(pts[2]-timeRange[0])),:]

		dividingImg=traceOutlines([pts],dist,(250,1200))
		
		matchingImg=traceOutlines(nearPts,1,(250,1200))	

		areaList=labelOverlap(dividingImg,matchingImg)
		if np.sum(areaList)==0:
			areaList=areaList+0
		elif areaList.shape[0]==1:
			areaList[0][0]=pts[3]
			areaList[0][1]=nearPts[areaList[0][1]-1,3]		
		else:
			for i in range(len(areaList)):
				areaList[i,0]=pts[3]
				areaList[i,1]=nearPts[areaList[i,1]-1,3]

		matchList=np.vstack((matchList,areaList))

	return matchList

def matchTracks(inList1,inList2,dist,timeRange):
	'''
	Find the closest pairs between inList1 and inList2.
	Normally, inList1 is masterEndList or the list of division events 
	and  inlist2 is masterStartList timeRange is the time before and 
	after inList1 where you should look for matching candidates. 
	This returns an array with [id1, id2, time, distance]
	'''
	matchList=np.zeros((1,4))
	for pts in inList1:	
		#Go over each track ends	
		
		#Generate bounding box
		boxPts=getBoundingBox(pts,dist)	
		boxPath=pa.Path(boxPts.copy())

		#Find starting points that fall inside the bounding box
		nearPts=inList2[(inList2[:,2]<=(pts[2]+timeRange[1])) & 
				(inList2[:,2]>=(pts[2]-timeRange[0])),:]

		centerPt=nearPts[:,0:2]
		
		pole1=np.transpose(np.array([nearPts[:,0] + nearPts[:,4]/2*np.sin(nearPts[:,6]*np.pi/180),
                                         nearPts[:,1] + nearPts[:,5]/2*np.cos(nearPts[:,6]*np.pi/180)]))
		pole2=np.transpose(np.array([nearPts[:,0] - nearPts[:,4]/2*np.sin(nearPts[:,6]*np.pi/180),
                                         nearPts[:,1] - nearPts[:,5]/2*np.cos(nearPts[:,6]*np.pi/180)]))

		inIndxC=boxPath.contains_points(centerPt)	
		inIndxL=boxPath.contains_points(pole1)
		inIndxR=boxPath.contains_points(pole2) 
		
		
		inIndx=inIndxC+inIndxL+inIndxR

		if inIndx.any():
			inPts=nearPts[inIndx,:]
			
			pole1=np.array([[pts[0] + pts[4]/2*np.sin(pts[6]*np.pi/180),
				         pts[1] + pts[5]/2*np.cos(pts[6]*np.pi/180)]])
			pole2=np.array([[pts[0] - pts[4]/2*np.sin(pts[6]*np.pi/180),
					 pts[1] - pts[5]/2*np.cos(pts[6]*np.pi/180)]])
			poleIn1=np.array([[inPts[:,0] + inPts[:,4]/2*np.sin(inPts[:,6]*np.pi/180),
				           inPts[:,1] + inPts[:,5]/2*np.cos(inPts[:,6]*np.pi/180)]])
			poleIn2=np.array([[inPts[:,0] - inPts[:,4]/2*np.sin(inPts[:,6]*np.pi/180),
				           inPts[:,1] - inPts[:,5]/2*np.cos(inPts[:,6]*np.pi/180)]])
	
				
			#Compute the distance between each one and the track	
			

			angleDot=10*(1-np.cos(np.abs(pts[6]-inPts[:,6])*np.pi/180))+1
			
			VL=eucledianDistances(pole1,poleIn1)
			VL=angleDot*VL[0][0]
			VR=eucledianDistances(pole2,poleIn2)
			VR=angleDot*VR[0][0]
			VC=eucledianDistances(np.array([pts[0:2]]),inPts[:,0:2])	
			VC=angleDot*VC[0]
			
			V21=eucledianDistances(pole2,poleIn1)
			V21=angleDot*V21[0][0]
			V12=eucledianDistances(pole1,poleIn2)
			V12=angleDot*V12[0][0]

			T=10*abs(inPts[:,2]-pts[2]-1)
			#Put the possible candidates into an array		
			for id in range(len(VL)):
				V=np.array([VL[id],VR[id],VC[id],V12[id],V21[id]])
				Dist=np.array([[pts[3],inPts[id,3],
					     pts[2],T[id]+V.min()]]) 			
				matchList=np.append(matchList,Dist,0)

	matchList[matchList[:,0]==matchList[:,1],:]=0	
	matchList=matchList[matchList[:,0]>0,:]

	return matchList

def splitIntoList(listIn,dim):
	'''
	listOut=splitIntoList(listIn,dim)
	This function splits an array according to a specific index. 
	For instance, if dim=1 is the time label, it will create a list 
	where list[t] is the data at time t.
	'''
	#Declare how fine the array will be split into
	divL=100
	#Initialize the first slice
	listOut=list(range(1+int(np.max(listIn[:,dim]))))
	listT=listIn[listIn[:,dim]<=divL,:]
	for id in np.arange(len(listOut)):
		if listT.shape[0]>0:
			listOut[int(id)]=(listT[listT[:,dim]==id,:])
		else:
			listOut[int(id)]=[]
		if np.mod(id,divL)==0:
			listT=listIn[(listIn[:,dim]>id) & 
				     (listIn[:,dim]<=(id+divL)),:]
		
	return listOut	

def revertListIntoArray(listIn):
	'''
	arrayOut=revertListIntoArray(listIn,dim)
	This function is the reverse of splitIntoList: it takes
	a list and concatenate the results into an array.
	'''
	#Declare the rough size of the array
	nID=100	

	arrayOut=list(range(len(listIn)/nID+1))
	k=0
	while len(listIn[k])<=0:
		k+=1
	arrayOut[0]=listIn[k].copy()
	
	k=0
	for id in range(2,len(listIn)):
		if (np.size(arrayOut[k])==1)and(np.size(listIn[id])>1):
			arrayOut[k]=listIn[id]
		elif isinstance(listIn[id],int):
			a=1
		elif len(listIn[id])>0:
			arrayOut[k]=np.vstack((arrayOut[k],listIn[id]))
		if np.mod(id,nID)==0:
			k=k+1

	arrayOutAll=arrayOut[0].copy()
	for id in range(1,len(arrayOut)):
		if np.size(arrayOut[id])>1:
			arrayOutAll=np.vstack((arrayOutAll,arrayOut[id]))	


	return arrayOutAll

def findDivisionEvents(trIn):
	'''
	divE=findDivisionEvents(trIn) goes through each cell tracks and 
	finds the times at which each divides.
	divE returns an array with divE=[cellID,time]
	'''
	trT=splitIntoList(trIn,3)
	divE=np.zeros((1,2))
	for tr in trT:
		if np.size(tr)>1:
			if len(tr)>10:
				divEvents=findDivs(tr[:,4])
				if divEvents.any():
					divTimes=tr[divEvents.tolist(),2]
					divT=np.vstack([divTimes*0 + tr[0,3],
						        divTimes])	
					divE=np.append(divE,divT.transpose(),0)
	divE=divE[1:len(divE),:]
	return divE

def findDivs(L):
	'''
	divTimes=findDivs(L) finds the index of the division events.
	L is the length of the cell
	'''	
	divTimes=np.array([])
	divJumpSize=17.5
	minSize=10
	std_thres=0.15
		#Remove the points with a large positive derivative, do this twice
	Lup=removePeaks(L.copy(),'Up')	
	Ldown=removePeaks(L.copy(),'Down')

	if np.sum(np.diff(Lup)**2)<np.sum(np.diff(Ldown)**2):	
		L=Lup.copy()
	elif np.sum(np.diff(Lup)**2)>np.sum(np.diff(Ldown)**2):
		L=Ldown.copy()
#		L=Ldown.copy()
	L=removePlateau(Ldown.copy())

	#Find the general location of a division event.
	divLoc=((np.diff((L))<-divJumpSize)&(L[:-1]>minSize)).nonzero()

	#Check if length is higher later
	divTimes=np.array(divLoc[0])
	for i in range(len(divTimes)):
		divs=divTimes[i]
		if L[divs:(divs+3)].max()>L[divs]:
			divTimes[i]=0
			
	divTimes=divLoc[0]+1.
	divTimes=divTimes[divTimes!=0]
	divTimes=divTimes[divTimes>3]

	return divTimes


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def removePeaks(Lin,mode='Down'):
	'''
	This function removes the peaks in a function.
	Peaks are defined as a sudden increase followed by a 
	sudden decrease in value within 4 datapoints. 
	'''
	divJumpSize=10
	LL=Lin.copy()
	#Find location of sudden jumps (up or down)
	jumpIDup=(np.diff(LL)>divJumpSize).nonzero()[0]
	jumpIDdown=(np.diff(LL)<-divJumpSize).nonzero()[0]


	if mode=='Down':	
		for id in jumpIDdown:
			if ((id-jumpIDup)==-1).nonzero()[0].size:
				LL[id+1]=LL[id].copy()
			elif ((id-jumpIDup)==-2).nonzero()[0].size:
				LL[id+1]=LL[id].copy() 
				LL[id+2]=LL[id].copy() 	
			elif ((id-jumpIDup)==-3).nonzero()[0].size:
				LL[id+1]=LL[id].copy()
				LL[id+2]=LL[id].copy()
				LL[id+3]=LL[id].copy()
		jumpIDup=(np.diff(LL)>divJumpSize).nonzero()[0]
		jumpIDdown=(np.diff(LL)<-divJumpSize).nonzero()[0]
	
                for id in jumpIDdown:
                       if ((id-jumpIDup)==1).nonzero()[0].size:
        	               LL[id]=LL[id+1].copy()
                       elif ((id-jumpIDup)==2).nonzero()[0].size:
                               LL[id-1]=LL[id+1].copy()
                               LL[id]=LL[id+1].copy()
                       elif ((id-jumpIDup)==3).nonzero()[0].size:
                               LL[id-2]=LL[id+1].copy()
                               LL[id-1]=LL[id+1].copy()
                               LL[id]=LL[id+1].copy()

	elif mode=='Up':
		for id in jumpIDup:
			if ((id-jumpIDdown)==-1).nonzero()[0].size:
				LL[id+1]=(LL[id]+LL[id+2])/2.
			elif ((id-jumpIDdown)==-2).nonzero()[0].size:
				LL[id+1]=2*LL[id]/3.+LL[id+3]/3.
				LL[id+2]=LL[id]/3.+2*LL[id+3]/3.
                        elif ((id-jumpIDdown)==-3).nonzero()[0].size:
                                LL[id+1]=3*LL[id]/4.+LL[id+4]/4.
                                LL[id+2]=LL[id]/2.+LL[id+4]/2.
                                LL[id+3]=LL[id]/4.+3*LL[id+4]/4.
                jumpIDup=(np.diff(LL)>divJumpSize).nonzero()[0]
                jumpIDdown=(np.diff(LL)<-divJumpSize).nonzero()[0]
        
                for id in jumpIDup:
                       if ((id-jumpIDdown)==1).nonzero()[0].size:
                               LL[id]=(LL[id-1]+LL[id+1])/2.
                       elif ((id-jumpIDdown)==2).nonzero()[0].size:
                               LL[id-1]=2*LL[id-2]/3.+LL[id+1]/3.      
                               LL[id]=LL[id-2]/3.+2*LL[id+1]/3.        
                       elif ((id-jumpIDdown)==3).nonzero()[0].size:
                               LL[id-2]=3*LL[id-3]/4.+LL[id+1]/4.
                               LL[id-1]=LL[id-3]/2.+LL[id+1]/2.
                               LL[id]=LL[id-3]/4.+3*LL[id+1]/4.

	return LL
	
def removePlateau(Lin):
	'''
	This function removes the plateaux in a function.
	A plateau is defined as a sudden increase (decrease) in value
	followed by a sudden decrease (increase). It is a plateau if the function
	becomes continuous once a constant value is added (substracted) to the plateau. 
	'''

	divJumpSize=10
	LL=Lin.copy()

	#Find the location of the sudden jumps (up or down)
	jumpIDup=(np.diff(LL)>divJumpSize).nonzero()[0]
	jumpIDup=jumpIDup[jumpIDup>0]
	jumpIDup=jumpIDup[jumpIDup<(len(LL)-1)]
	
	jumpIDdown=(np.diff(LL)<-divJumpSize).nonzero()[0]
	jumpIDdown=jumpIDdown[jumpIDdown>0]
	jumpIDdown=jumpIDdown[jumpIDdown<(len(LL)-1)]

	for id in jumpIDup:
		nextDown=-1
		previousDown=-1
		if jumpIDdown[jumpIDdown>id].any():
			nextDown=jumpIDdown[jumpIDdown>id][0]
		if jumpIDdown[jumpIDdown<=id].any():
			previousDown=jumpIDdown[jumpIDdown<=id][-1]

		if nextDown>0:
			jumpDiffDown=np.abs(np.abs(LL[id]-LL[id+1])-np.abs(LL[nextDown]-LL[nextDown+1]))
		else:
			jumpDiffDown=np.Inf
		if previousDown>0:
			jumpDiffUp=np.abs(np.abs(LL[id]-LL[id+1])-np.abs(LL[previousDown]-LL[previousDown+1]))
		else:
			jumpDiffUp=np.Inf	
		if jumpDiffDown<jumpDiffUp:
			LL[(id+1):(nextDown+1)]=LL[(id+1):(nextDown+1)]-(LL[id+1]-LL[id])
		elif jumpDiffUp<jumpDiffDown:
			LL[(previousDown+1):(id+1)]=LL[(previousDown+1):(id+1)]+(LL[id+1]-LL[id])
	return LL


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)	

def findDD(trIn):
	divData=findDivisionEvents(trIn)
	return divData

def splitTracks(trIn):
	'''
	This function splits each tracks into cell segment that start with birth
	and ends with cell divisions or device escape. 
	'''
	trIn=np.hstack((trIn,np.zeros((len(trIn),2))))
	trI=splitIntoList(trIn,3)
	divData=findDivisionEvents(trIn.copy()) 
	x=np.array([0,0])
	k=1
	idList=np.unique(trIn[:,3]).astype('int')
	for id in idList:
		if np.intersect1d(divData[:,0],np.array(id)):
			divList=divData[divData[:,0]==id,:]
			x[0]=trI[id][0,2]
			for pt in divList:
				x[1]=pt[1].copy()
				if (pt[1]!=divList[0][1]):
					trI[id][(trI[id][:,2]==x[0]),8]=k-1
					trI[id][(trI[id][:,2]==x[0]),9]=k-1
				trI[id][(trI[id][:,2]==(x[1]-1)),8]=k+1
				trI[id][(trI[id][:,2]>=x[0])&(trI[id][:,2]<x[1]),3]=k
				k=k+1
				x[0]=x[1].copy()
			trI[id][(trI[id][:,2]==x[0]),8]=k-1
			trI[id][(trI[id][:,2]==x[0]),9]=k-1
			trI[id][trI[id][:,2]>=x[0],3]=k
			k=k+1
		else:
			trI[id][:,3]=k
			k=k+1	

	trOut=revertListIntoArray(trI)		
#	trOut=removeShortCells(trOut,20)	
	
	return trOut

def findDaughters(trIn,merge=False):
	'''
	
	'''
	tr=trIn.copy()

	meanL=np.mean(tr[:,4])
	meanW=np.mean(tr[:,5])

	tr[tr[:,4]<meanL,4]=meanL
	tr[tr[:,5]<meanW,5]=meanW
	#Find and match division events
	if len(trIn[0,:])>8:	
		inList0=tr[(tr[:,8]==tr[:,9])&(tr[:,8]>0),:]
		inList1=tr[(tr[:,8]>0)&(tr[:,9]==0),:]	
		mList=matchTracksOverlap(inList1,inList0,2,[0,2])
		divData0=matchIndices(mList,'Area')	
	else:
		divData0=np.zeros((1,3))	
	tr0=tr.copy()
	mList0=fixTracks(tr0,2.5,[2,6])
	#if np.sum(mList0[:,3]<meanL):	
#		mList1=mList0[mList0[:,3]<meanL,:]
#	else:
	mList1=mList0.copy()
	divData1=matchIndices(mList1,'Area')
	for dd in divData0:
		divData1[divData1[:,1]==dd[1],:]=0
		divData1[divData1[:,0]==dd[0],:]=0

	divData1=divData1[divData1[:,0]>0,:]

	mList2=mList0.copy()
	for div in divData0:
		mList2[(mList0[:,1]==div[1]),:]=0
	for div in divData1:
		mList2[(mList0[:,1]==div[1]),:]=0

	mList2=mList2[mList2[:,0]>0,:]

	divData2=matchIndices(mList2,'Area')
	
	divData=np.vstack((divData0,divData1,divData2))
	divData=divData[divData[:,0].argsort(),]
	divData=unique_rows(divData)

	return divData

def mergeIndividualTracks(trIn,divData):
	'''
	Merge tracks from the index list divData. Only merge if data is continuous
	or cell is larger in the near future.
	'''
	trT=splitIntoList(trIn.copy(),3)
	idList=np.unique(divData[:,0])
	for k in range(len(idList)):
		motherCell=int(idList[k])
		daughterCell=divData[divData[:,0]==motherCell,1].astype('int')
		if len(daughterCell)==1:
			trT[motherCell]=np.vstack((trT[motherCell],trT[daughterCell]))
			trT[motherCell][:,3]=motherCell
			trT[daughterCell]=[]
			idList[idList==daughterCell]=motherCell
			divData[divData[:,0]==motherCell,0]=0
			divData[divData[:,0]==daughterCell,0]=motherCell
	trOut=revertListIntoArray(trT)
	
        trOut=trOut[trOut[:,2].argsort(-1,'mergesort'),]
        trOut=trOut[trOut[:,3].argsort(-1,'mergesort'),]
                                
        trOut[(np.diff(trOut[:,2])==0)&(np.diff(trOut[:,3])==0),:]=0
        trOut=trOut[trOut[:,0]>0,:]
	
	return trOut

def mergeTracks(trIn):
	'''

	'''
	#Fix tracks that skip frames without divisions
	#trOut=removeShortTracks(trIn,1)
	trOut=trIn.copy()
	
	#Get division data
	divData=findDaughters(trOut)
	
	masterEndList=trOut[np.diff(trOut[:,3])>0,:]
	for i in np.unique(masterEndList[:,3]):
		dd=divData[divData[:,0]==i,:]
		if len(dd)>1:
			masterEndList[masterEndList[:,3]==i,:]=0
	masterEndList=masterEndList[masterEndList[:,0]>0,:]
	
	orphanCoordinates=np.zeros((1,trOut.shape[1]))

	for i in np.unique(trOut[trOut[:,2]>50,3]):
		dd=divData[divData[:,1]==i,:]
		if len(dd)<1:
			orphanCoordinates=np.vstack((orphanCoordinates,trOut[trOut[:,3]==i,:][0]))

	#Link between orphan tracks at birth and track ends
	mList=matchTracksOverlap(masterEndList,orphanCoordinates,4,[3,8])
	mList=mList[mList[:,0]>0,:]
	if mList.any():
		matchData=matchIndices(mList,'Area')
	else:
		matchData=np.zeros((3,1))


	trT=splitIntoList(trOut,3)
	ids=np.unique(trOut[:,3])
	ids=np.flipud(ids)
	for k in range(len(ids)):
		i=ids[k]
		divD=divData[divData[:,0]==i,:]
		mData=matchData[matchData[:,0]==i,:]
		if len(divD)==2:
			trT[int(i)][-1,8]=divD[0,1]
			trT[int(divD[0,1])][0,8]=i
			trT[int(divD[0,1])][0,9]=i
			trT[int(i)][-1,9]=divD[1,1]
			trT[int(divD[1,1])][0,8]=i
			trT[int(divD[1,1])][0,9]=i
		elif (len(mData)==1)and(len(divD)==1):
			trT[int(i)][-1,8]=divD[0,1]
			trT[int(divD[0,1])][0,8]=i
			trT[int(divD[0,1])][0,9]=i
			trT[int(i)][-1,9]=mData[0,1]
			trT[int(mData[0,1])][0,8]=i
			trT[int(mData[0,1])][0,9]=i
		elif (len(mData)==1)and(len(divD)==0):
			if checkContinuous(trT,int(i),int(mData[0,1])):
				trT[int(i)][:,3]=mData[0,1]
				trT[int(i)][1:,8:10]=0
				trT[int(mData[0,1])][:-1,8:10]=0
				trT[int(mData[0,1])]=np.vstack((trT[int(i)],trT[int(mData[0,1])]))
#				trT[int(mData[0,1])]=[]
				divData[divData[:,0]==i,0]=mData[0,1]
				divData[divData[:,1]==i,1]=mData[0,1]
				matchData[matchData[:,0]==i,0]=mData[0,1]
				matchData[matchData[:,1]==i,1]=mData[0,1]
#				ids[ids==i]=mData[0,1]
			else:	
				trT[int(i)][-1,8]=mData[0,1]
				trT[int(i)][-1,9]=mData[0,1]
				trT[int(mData[0,1])][0,8]=i
				trT[int(mData[0,1])][0,9]=i
		elif (len(mData)==0)and(len(divD)==1):	
			if checkContinuous(trT,int(i),int(divD[0,1])):
				trT[int(i)][:,3]=divD[0,1]
				trT[int(i)][1:,8:10]=0
				trT[int(divD[0,1])][:-1,8:10]=0
				trT[int(divD[0,1])]=np.vstack((trT[int(i)],trT[int(divD[0,1])]))
#				trT[int(divD[0,1])]=[]
				divData[divData[:,0]==i,0]=divD[0,1]
				divData[divData[:,1]==i,1]=divD[0,1]
				matchData[matchData[:,0]==i,0]=divD[0,1]
				matchData[matchData[:,1]==i,1]=divD[0,1]
#				ids[ids==i]=divD[0,1]
			else:	
				trT[int(i)][-1,8]=divD[0,1]
				trT[int(i)][-1,9]=divD[0,1]
				trT[int(divD[0,1])][0,8]=i
				trT[int(divD[0,1])][0,9]=i

	trOut=revertListIntoArray(trT)

	trOut=trOut[trOut[:,2].argsort(-1,'mergesort'),]
	trOut=trOut[trOut[:,3].argsort(-1,'mergesort'),]

	trOut[(np.diff(trOut[:,2])==0)&(np.diff(trOut[:,3])==0),:]=0
	trOut=trOut[trOut[:,0]>0,:]


	

	return trOut


def checkContinuous(trIn,id1,id2):
	'''

	'''
	dT1=trIn[id1].copy()
	dT2=trIn[id2].copy()

	L=np.hstack((dT1[:,4],dT2[:,4]))
	if len(L)>10:
		divs=findDivs(L)
	else:
		return True
	if divs.any():
		return False
	else:
		return True


	
def renameTracks(trIn):
	'''
	This script changes the track assignment list to remove
	tracks which are childless orphans. 	
	'''
	#trIn=removeShortTracks(trIn,0)
	trI=splitIntoList(trIn,3)
	trOut=np.zeros((1,9))
	dT2=np.zeros((1,9))
	k=0
	#Find ids of cells that divide
	divID=np.unique(np.abs(trIn[trIn[:,8]!=0,3]))
	IdNum=0
	newIndex=np.zeros((np.max(trIn[:,3]),1))
	for id in np.unique(divID):
		IdNum=IdNum+1
		#relabel track
		newIndex[IdNum,0]=id
		dT=trI[int(id)]
		#find number of divisions		
		nDiv=(dT[:,8]>0).nonzero()[0]
		#Loop over each division
		for i in range(len(nDiv)):
			nD=(divID==dT[nDiv[i],8]).nonzero()[0]
			if nD.any(): #if daughter also divides, do not touch it
				dT[nDiv[i],8]=nD[0]+1
			else: #If it didn't divide, relabel it
				k=k+1
				dId=dT[nDiv[i],8]
				newID=len(divID)+k
				newIndex[newID]=dId
	trOut=trIn.copy()
	trOut[:,3]=0
	for id in range(len(newIndex)):
		if not newIndex[id]==0:		
			trOut[trIn[:,3]==newIndex[id],3]=id
			trOut[trIn[:,8]==newIndex[id],8]=id
			trOut[trIn[:,8]==-newIndex[id],8]=-id
	trOut=trOut[trOut[:,3]>0,:]

        trOut=trOut[trOut[:,2].argsort(-1,'mergesort'),]
        trOut=trOut[trOut[:,3].argsort(-1,'mergesort'),]

	#Remove multiply assigned cells	
#	trOut=trOut[np.diff(trOut[:,2])>0,:]

	return trOut

def findFamilyID(trIn):
	"""
	Label the cells that are member of the same family with their 
	family ID in the 10th column.
	"""
	trIn=np.hstack([trIn,np.zeros((len(trIn),1))])
	trI=splitIntoList(trIn,3)
	cellIdList=np.unique(trIn[:,3])
	famID=-1
	k=0
	stop=False		
	
	
	while len(cellIdList)>0:		
		famID=famID+1
		unlabelledCell=cellIdList[0]
		famList=np.array([unlabelledCell])
		while not stop:
			famList1=famList.copy()
			for id in famList:
				if len(trI[int(id)])>0:
					famList0=np.unique(trI[int(id)][-1,8:10])
					famList=np.unique(np.hstack([famList,
					                     famList0]))
			if np.array_equal(famList,famList1):
				stop=True	
		for id in famList:
			trI[int(id)][:,-1]=famID
		cellIdList=np.setdiff1d(cellIdList,famList)
		stop=False
	trLong=revertListIntoArray(trI)	
	
	return trLong
		
def matchFamilies(trIn):
	'''
	This script matches the start of a new family with 
	the closest cell.	
	'''
	trIn=np.hstack([trIn,np.zeros((len(trIn),1))])	

	freeID=np.setdiff1d(np.arange(2*np.max(trIn[:,3])),np.unique(trIn[:,3]))
	freeID=freeID[1:].copy()
	trIn=np.vstack((trIn,trIn[0]))
	trIn[-1,:]=0
	trIn[-1,3]=2*np.max(trIn[:,3])

	trFam=splitIntoList(trIn,10)
	trI=splitIntoList(trIn,3)	
	
	famStart=trIn[0]
	for tt in trFam:
		if len(tt[:,0])>10:
			famStart=np.vstack((famStart,tt[0]))
			
	famStart[:,5]=famStart[:,5]/2
	initialIDs=np.unique(trIn[trIn[:,2]<25,10])
	for i in initialIDs:
		famStart[famStart[:,10]==i,:]=0
	famStart=famStart[famStart[:,0]>0,:]

	listOfMatches=matchTracksOverlap(famStart,trIn,2.5,[4,-1])	

	matchData=matchIndices(listOfMatches,'Area')
#	return famStart,matchData
	for md in matchData:
		trI[int(md[0])][0,8:10]=md[1]
		
		divTime=trI[int(md[0])][0,2]
		if len(trI[int(md[1])][trI[int(md[1])][:,2]<divTime,3])==0:
			divTime+=2
		trI[int(md[1])][trI[int(md[1])][:,2]>=divTime,3]=freeID[0]
		trI[int(md[1])][trI[int(md[1])][:,2]==(divTime),8]=md[1]
		trI[int(md[1])][trI[int(md[1])][:,2]==(divTime),9]=md[1]		
		trI[int(md[1])][trI[int(md[1])][:,2]==(divTime-1),8]=md[0]
		trI[int(md[1])][trI[int(md[1])][:,2]==(divTime-1),9]=freeID[0]		

		daughterID1=trI[int(md[1])][-1,8]
		daughterID2=trI[int(md[1])][-1,9]
		print md[1],daughterID1,daughterID2,freeID[0]
		if (daughterID1>0)&(daughterID1<=len(trI)):
			if len(trI[int(daughterID1)]):
				trI[int(daughterID1)][0,8:10]=freeID[0]
		if (daughterID2>0)&(daughterID2<=len(trI)):
			if len(trI[int(daughterID2)]):
				trI[int(daughterID2)][0,8:10]=freeID[0]

		freeID=freeID[1:].copy()

	trOut=revertListIntoArray(trI)

        trOut=trOut[trOut[:,2].argsort(-1,'mergesort'),]
        trOut=trOut[trOut[:,3].argsort(-1,'mergesort'),]
        
        trOut[(np.diff(trOut[:,2])==0)&(np.diff(trOut[:,3])==0),:]=0
        trOut=trOut[trOut[:,0]>0,:]

	trOut=findFamilyID(trOut[:,0:10])	

	return trOut

	
def fixFamilies(trIn):
	'''
	Sometimes, matchFamilies cannot assign a division event because it can't find
	the mother cell at the time of the division. 
	This script goes through every family that should be matched and assigns the 
	division id to the mother cell.
	'''
	
	for famId in range(1,int(max(trIn[:,9]))):
		dT=trIn[trIn[:,9]==famId,:]
		if dT.shape[0]:
			daughterID=dT[:,3][0]
			if dT[0,8]<0:
				motherID=abs(dT[0,8])
				divPos=((trIn[:,3]==motherID).nonzero()[0])
				if trIn[divPos[-1],8]==0:
					trIn[divPos[-1],8]=daughterID
				else:
					trIn[divPos[-2],8]=daughterID

	trOut=findFamilyID(trIn[:,0:9])
	return trOut

def addCellAge(trIn):
	'''
	Adds the age of the cells in the last column.
	It also fills in the blank and missing data (eg. when track skip a time)
	'''	

	trIn=np.hstack((trIn,np.zeros((len(trIn),1))))
	trI=splitIntoList(trIn,3)

	for i in range(len(trI)):
		if len(trI[i])>0:
			motherID=trI[i][0,8]
			if motherID>0:
				if len(trI[int(motherID)])>0:
					divisionTime=trI[int(motherID)][-1,2]
			else:
				divisionTime=trI[i][0,2]	
			tStart=trI[i][0,2]-divisionTime-1
			trI[i][0,-1]=np.max((tStart,0))
			trI[i][1:,-1]=trI[i][0,-1]+np.cumsum(np.diff(trI[i][:,2]))
			if trI[i][0,-1]>0:
				trI[i]=np.vstack((-1*np.ones((tStart,trI[i].shape[1])),trI[i]))
				trI[i][:tStart,0]=trI[i][tStart,0]
				trI[i][:tStart,1]=trI[i][tStart,1]
				trI[i][:tStart,2]=trI[i][tStart,2]-np.arange(tStart,0,-1)
				trI[i][:,3]=i
				trI[i][:tStart,4]=trI[i][tStart,4]
				trI[i][:tStart,5]=trI[i][tStart,5]
				trI[i][:tStart,6]=trI[i][tStart,6]
				trI[i][:tStart,7]=trI[i][tStart,7]
				trI[i][0,8]=trI[i][tStart,8]
				trI[i][0,9]=trI[i][tStart,9]
				trI[i][1:-1,8]=0
				trI[i][1:-1,9]=0
				trI[i][:,10]=trI[i][-1,10]
				trI[i][:tStart,-1]=np.arange(tStart)
			
			if (np.diff(trI[i][:,2])>1).any():
				trI[i]=fillGap(trI[i].copy())

	trOut=revertListIntoArray(trI)
        trOut[(np.diff(trOut[:,2])==0)&(np.diff(trOut[:,3])==0),:]==0
	trOut=trOut[trOut[:,0]>0,:]

	return trOut

def fillGap(tr):
	'''

	'''
	trIn=tr.copy()
	stop=False
	k=0
	while stop==False:
		dt=trIn[k+1,2]-trIn[k,2]
		if dt>1:
			dTemp=trIn[k,:].copy()
			dTemp[2]=trIn[k,2]+1
			dTemp[11]=trIn[k,11]+1
			trIn=np.insert(trIn,k+1,dTemp,0)
			k=k+1
		if dt>2:
			dTemp=trIn[k,:].copy()
			dTemp[2]=trIn[k,2]+1
			dTemp[11]=trIn[k,11]+1
			trIn=np.insert(trIn,k+1,dTemp,0)
			k=k+1
		elif dt>3:
			dTemp=trIn[k,:].copy()
			dTemp[2]=trIn[k,2]+1
			dTemp[-1]=trIn[k,-1]+1
			trIn=np.insert(trIn,k+1,dTemp,0)
			k=k+1
		elif dt>4:
			print 'bar'

		if (k+2)==len(trIn):
			stop=True
		else:
			k=k+1

	return trIn

def addElongationRate(trIn):
	'''
	Fits an exponential to the length between every divisions to find the elongation rate.
	'''
	trIn=np.hstack((trIn,np.zeros((len(trIn),1))))
	trI=splitIntoList(trIn,3)

	for id in np.unique(trIn[:,3]):
		dT=trI[int(id)].copy()
		if len(dT)>15:
			dT=dT[5:-5,:]
			z = np.polyfit(range(len(dT)), np.log(np.abs(dT[:,4])), 1)
			dT[:,-1]=z[0]
			trI[int(id)][:,-1]=z[0]
	trOut=revertListIntoArray(trI)
	
	
	return trOut

def smoothLength(trIn):
	'''
	Goes through each track and applies a removePeaks to the length
	'''
	
	trI=splitIntoList(trIn,3)

	for id in range(len(trI)):
		dT=trI[id]
		if len(dT):
			LL=removePeaks(dT[:,4],'Down')
			LL=removePlateau(LL)
			trI[id][:,4]=LL.copy()

	trOut=revertListIntoArray(trI)
	return trOut
	

def pathologicalTracks(trIn):
	'''
	Extracks the tracks that either are born without mothers or die without children
	'''

	trI=splitIntoList(trIn,3)

	for id in range(len(trI)):
		if len(trI[int(id)]):
			if (trI[int(id)][0,8]!=0)&(trI[int(id)][-1,8]!=0):
				trI[int(id)][:,3]=0
	trOut=revertListIntoArray(trI)

	trOut=trOut[trOut[:,3]>0,:]

	return trOut


'''
def computeOpticalFlow:
	for i in range(800):
		bar0=array(im)
		im.seek(i)
	    	bar1=array(im)
		A = cv2.calcOpticalFlowFarneback(uint16(bar0),uint16(bar1),
						 None,pyr_scale=0.5,levels=1,
						 winsize=25,iterations=1,
						 poly_n=5,poly_sigma=1.1,
						 flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
		xx[i]=sum(A[0:700,:,1])
		print i
'''

'''
	xx=zeros((600,1))
	for i in range(350):
    	bar0=cv2.transpose(cv2.imread('Fluo_scan_'+str(i)+'_Pos0_g.tif',-1))
    	bar1=cv2.transpose(cv2.imread('Fluo_scan_'+str(i+1)+'_Pos0_g.tif',-1))
    	A = cv2.calcOpticalFlowFarneback(uint8(bar0),uint8(bar1),
					 pyr_scale=0.5,levels=1,winsize=25,
					 iterations=1,poly_n=5,poly_sigma=1.1,
					 flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    	xx[i]=sum(A[:,:,0])
    	print i
'''
def optimizeParameters(fPath,num):
	"""
	Tests the processing parameters on a test file 
	"""
	import random as rnd
	lnoise0=1
	lobject0=8
	thres0=2
	tifList=[]
	fileList=sort_nicely(os.listdir(fPath))
	for file in fileList:
		if file.endswith('tif'):
			tifList.append(file)
	if num==0:
		num=rnd.randint(1,len(tifList))
		print num
	stop=False	
	while not stop:
		lnoise0=(raw_input('What is the typical noise size you want to remove (leave empty for current='+str(lnoise0)+')?') or lnoise0)	
		lobject0=(raw_input('What is the typical object size (leave empty for current='+str(lobject0)+')?') or lobject0)
		thres0=(raw_input('What is the intensity threshold (leave empty for current='+str(thres0)+')?') or thres0)
		print 'Please examine the resulting image, close it when done'	
		img=cv.imread(fPath+tifList[num],-1)
		img=cv.transpose(img)
		bwImg=processImage(img,scaleFact=1,sBlur=0,sAmount=0,
				   lnoise=lnoise0,lobject=lobject0,
				   thres=np.double(thres0),solidThres=0.65,
				   lengthThres=1.5)
		fig = plt.figure()
		ax1 = fig.add_subplot(3,1,1)
		ax1.imshow(img)	
		ax1.set_title('Raw Image')	
		
		ax2 = fig.add_subplot(3,1,2)	
		ax2.imshow(bpass(img,lnoise0,lobject0))	
		ax2.set_title('Filtered image. Noise size='+str(lnoise0)+'. Object size='+str(lobject0)+'. ')	
	
		ax3 = fig.add_subplot(3,1,3)	
		ax3.imshow(bwImg)
		ax3.set_title('Thresholded image. Threshold value='+str(thres0)+'. ')	
		
		plt.show()
		satisfiedYN=raw_input('Are you satisfied with the  parameters? (yes or no) ')
		if satisfiedYN=='yes':
			stop=True
			

	return lnoise0,lobject0,thres0	

def splitRegions(fPath,numberOfRegions):
	"""
	Performs a kmeans analysis to identify each regions. 
	"""
	import random as rnd
	import scipy.cluster.vq as vq
	lnoise0=1
	lobject0=8
	thres0=2
	tifList=[]
	fileList=sort_nicely(os.listdir(fPath))
	for file in fileList:
		if file.endswith('tif'):
			tifList.append(file)
	num=rnd.randint(1,int(len(tifList)/10.))
	print num
	img=cv.imread(fPath+tifList[num],-1)
	img=cv.transpose(img)
	bwImg=processImage(img,scaleFact=1,sBlur=0,sAmount=0,
			   lnoise=lnoise0,lobject=lobject0,
			   thres=np.double(thres0),solidThres=0.65,
			   lengthThres=1.5)
	posList=regionprops(bwImg)[:,0:2]
	wPosList=vq.whiten(posList)

	idList=vq.kmeans2(wPosList[:,1],numberOfRegions)	
	x=np.zeros((numberOfRegions,))
	for id in range(numberOfRegions):
		x[id]=np.mean(posList[idList[1]==id,1])	

	x=np.sort(x)
	xLimits=np.array([x[0]/2,(x[0]+x[1])/2])
	for id in range(1,numberOfRegions-1,1):
		xLimits=np.vstack((xLimits,np.array([(x[id-1]+x[id])/2,(x[id]+x[id+1])/2])))
	
	xLimits=np.vstack((xLimits,np.array([(x[id]+x[id+1])/2,(x[-1]+img.shape[0])/2])))
	
	return xLimits.astype(np.int)

def saveListOfArrays(fname,listData):
	'''
	Save a list of different length arrays (but same width) as a single array
	'''
	arrayOut=np.hstack((0*np.ones((len(listData[0]),1)),listData[0],))
	for i in range(1,len(listData)):
		arrayTemp=np.hstack((i*np.ones((len(listData[i]),1)),listData[i],))
		arrayOut=np.vstack((arrayOut,arrayTemp))

	np.savez(fname,arrayOut)

def loadListOfArrays(fname):
	'''
	Loads an array generated by saveListOfArrays and export it as a list of arrays
	'''
	arrayInObj=np.load(fname)
	arrayIn=arrayInObj['arr_0']
	listLength=np.max(arrayIn[:,0])+1

	listOut=[]

	for i in range(int(listLength)):
		listOut.append(arrayIn[arrayIn[:,0]==i,1:])

	return listOut

#When run directly inside a folder containing images of cells, this script creates the rawTrackData file.
if __name__ == "__main__":
	'''
	
	Options:
	-i, --interactive (default): ask for the various prompts
	-n, --no-interaction: processes the images and does the 
		track analysis in the current directory
	'''
	import sys
	import cv2 as cv
	import numpy as np
	
	INTERACTIVE=True

	for var in sys.argv:
		if (var == '-i'):
			INTERACTIVE=True
		elif (var == '--interactive'):
			INTERACTIVE=True
		elif (var == '-n'):
			INTERACTIVE=False
		elif (var=='--no-interaction'):
			INTERACTIVE=False
	
	
	lnoise,lobject,thres=1,8,2

	if INTERACTIVE:
		FILEPATH=(raw_input('Please enter the folder you want to analyze (leave empty to use current location) ') or './')
		PROCESSFILES=raw_input('Do you want to process the image files? (yes or no) ')	
		if PROCESSFILES=='yes':
			OPTIMIZEPROCESSING=raw_input('Do you want to optimize the image processing parameters? (yes or no) ')
			if OPTIMIZEPROCESSING=='yes':
				lnoise,lobject,thres=optimizeParameters(FILEPATH,0)
		MULTIPLEREGIONS=raw_input('Are there multiple disjointed regions in each image? (yes or no) ')
		if MULTIPLEREGIONS=='yes':
			NUMBEROFREGIONS=int(raw_input('How many regions per field of view? (Enter a number) '))
		else:
			NUMBEROFREGIONS=1
		LIMITFILES=raw_input('Enter the number of files to analyze (leave empty for all) ' or '0')
		if not LIMITFILES:
			LIMITFILES=0
		print LIMITFILES
		LINKTRACKS=raw_input('Do you want to link the cell tracks? (yes or no) ')
		if LINKTRACKS=='yes':
			PROCESSTRACKS=raw_input('Do you want to analyze the cell tracks? (yes or no) ')
		SAVEPATH=(raw_input('Please enter the location where the analyzed files will be saved (leave empty to use current location) ') or './')
	else:
		FILEPATH='./'
		PROCESSFILES='no'
		LINKTRACKS='yes'
		PROCESSTRACKS='yes'
		SAVEPATH='./'
		MULTIPLEREGIONS='no'
		NUMBEROFREGIONS=1		

	np.savez(SAVEPATH+'processFiles.npz',lnoise=lnoise,lobject=lobject,thres=thres)

	if MULTIPLEREGIONS=='yes':
		lims=splitRegions(FILEPATH,NUMBEROFREGIONS)
	else:
		lims=np.array([[0,-1]])
	if PROCESSFILES=='yes':
		masterL,LL,AA=trackCells(FILEPATH,np.double(lnoise),np.double(lobject),np.double(thres),lims,int(LIMITFILES))
		for id in range(NUMBEROFREGIONS):
			saveListOfArrays(SAVEPATH+'masterL_'+str(id)+'.npz',masterL[id])
			saveListOfArrays(SAVEPATH+'LL_'+str(id)+'.npz',LL[id])

	if LINKTRACKS=='yes':
		for id in range(NUMBEROFREGIONS):
			#masterL=loadListOfArrays(SAVEPATH+'masterL_'+str(id)+'.npz')
			#LL=loadListOfArrays(SAVEPATH+'LL_'+str(id)+'.npz')
			masterL=loadListOfArrays(SAVEPATH+'masterL.npz')
			LL=loadListOfArrays(SAVEPATH+'LL.npz')
			tr=linkTracks(masterL,LL)
			if PROCESSTRACKS=='yes':
				tr=processTracks(tr)
			with open(SAVEPATH+'/trData_'+str(id)+'.dat', 'wb') as f:	
				f.write(b'# xPos yPos time cellID cellLength cellWidth cellAngle avgIntensity divisionEvents familyID cellAge elongationRate\n')
				np.savetxt(f,tr)
			print 'The analysis of region '+str(id)+' is complete. Data saved as '+SAVEPATH+'trData_'+str(id)+'.dat'	
		
