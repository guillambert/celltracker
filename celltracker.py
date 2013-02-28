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
import matplotlib.pylab as plt
import os as os
import munkres as mk
import time as time
import sys as sys

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
	imgOut=filtered
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
	regionID=imgIn[seedPt[0],seedPt[1]]

	bwImg=imgIn.copy()
	bwImg[labelledImg==regionID]=pixelValue		

	return bwImg

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
	bwImg0=bwlabel(np.uint8(bwImg.copy()))
	bw=np.zeros(bwImg0.shape)
	csa,_ = cv.findContours(bwImg.copy(), 
				mode=cv.RETR_TREE, 
				method=cv.CHAIN_APPROX_SIMPLE)
	numC=int(len(csa))
	k=0
	for i in range(0,numC):
		if len(csa[i])>=5:
			k=k+1
	avgCellI=np.zeros((k,1),dtype=float)
	rawImg=rawImg.reshape(-1)
	k=0
	for i in range(0,numC):
		if len(csa[i])>=5:
			# Average Pixel value		
			regionMask = (bwImg0==(i))
			regionMask=regionMask.reshape(-1) 
			avgCellI[k]=np.mean(rawImg[regionMask>0])
			k=k+1
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
	dilImg=cv.dilate(imgIn,None,iterations=nIter)
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
						A[indx1][indx2]=1/matchData[i,2]
			#Find matching elements with munkres algorithm
			m=mk.Munkres()
			inds=m.compute(-A)
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

def putLabelOnImg(fPath,tr,dataRange,dim):
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
	if not dataRange:
		dataRange=range(len(tifList))
	for t in dataRange:
		time_start=time.time()
		fname = tifList[t] 
		print fPath+fname
                img=cv.imread(fPath+fname,-1)
                img=cv.transpose(img)
		bwImg=processImage(img,scaleFact=1,sBlur=0.5,
				   sAmount=0,lnoise=1,lobject=8,
				   thres=2,solidThres=0.65,
				   lengthThres=1.5,widthThres=20)
		plt.imshow(bwImg)
		plt.hold(True)
		trT=trTime[t]
		for cell in range(len(trT[:,3])):
			'''plt.text(trT[cell,0],trT[cell,1],
			            str(trT[cell,dim])+', '+str(trT[cell,3]),
				    color='w',fontsize=6)
			'''
			plt.text(trT[cell,0],trT[cell,1],
				 str(trT[cell,dim]),color='w',fontsize=10)
			boxPts=getBoundingBox(trT[cell,:])
			plt.plot(boxPts[:,0],boxPts[:,1],'w')
		plt.title(str(t))
		plt.hold(False)
		#plt.savefig(fPath+"Fig"+str(t)+".jpg",dpi=(120))
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
	#Segment Cells accorging to solidity
	bwImg=segmentCells(bwImg>0,regionprops(bwImg>0,scaleFact)[:,6],
			   solidThres,2)	
	#Segment cells according to length
	mLength=np.mean(regionprops(bwImg>0,scaleFact)[:,2])
	bwImg=segmentCells(bwImg>0,-regionprops(bwImg>0,scaleFact)[:,2],
			   -lengthThres*mLength,2)	
	return bwImg
	#Segment cells according to width (currently not used)
	bwImg=segmentCells(bwImg>0,-regionprops(bwImg>0,scaleFact)[:,3],
			   -widthThres,3)	
	bwImg=np.uint8((bwImg>0))
	
	return bwImg

def preProcessCyano(brightImg,chlorophyllImg):
	'''
	Pre-process the Chlorophyll and brightfield images so that they can be
	analyzed with processImages
	''' 
	solidThres=0.75
	cellMask = cv.dilate(np.uint8(bpass(chlorophyllImg,1,10)>10),None,iterations=15)
	processedBrightfield = bpass(brightImg,1,10)>250

	dilatedIm=cv.dilate(removeSmallBlobs(processedBrightfield*cellMask,15),None,iterations=2)
#	dilatedIm=(removeSmallBlobs(processedBrightfield*cellMask,15))

	if np.sum(cellMask==0):
		seedPt = ((1-cellMask).nonzero()[0][0],(1-cellMask).nonzero()[1][0])

		imgOut=dilateConnected(1-floodFill(dilatedIm,seedPt,1),2)
#		imgOut=(1-floodFill(dilatedIm,seedPt,1))
	else:
		imgOut=dilateConnected(1-dilatedIm,2)
	
	#Segment Cells accorging to solidity
#	imgOut=segmentCells(imgOut>0,regionprops(imgOut>0,1)[:,6],
#			   solidThres,2)	

	return np.uint8(imgOut)

def trackCells(fPath,lnoise=1,lobject=8,thres=3):
	'''
	This prepares the data necessary to track the cells.
	masterL,LL,AA=trackCells(fPath,lnoise=1,lobject=8,thres=5)
	processes the files in fPath.
	'''
	#Initialize variables
	sBlur=0.5
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
	kmax=len(tifList)
	startProgress('Analyzing files:')
	for fname in np.transpose(tifList):
#		print(fPath+fname)
		k=k+1
		progress(np.double(k)/np.double(kmax))
		img=cv.imread(fPath+fname,-1)
		img=cv.transpose(img)
		bwImg=processImage(img,scaleFact,sBlur,
				   sAmount,lnoise,lobject,
				   thres,solidThres,lengthThres)
		bwL=bwlabel(bwImg)
		fileNum=int(re.findall(r'\d+',fname)[0])
		if bwL.max()>5:
			regionP=regionprops(bwImg,scaleFact)
			#avgCellI=avgCellInt(img.copy(),bwImg)
			avgCellI=np.zeros((len(regionP),1))
			regionP=np.hstack([regionP,0*avgCellI])
			if (fileNum-fileNum0)==1:
				areaList=labelOverlap(bwL0,bwL)
				AA.append(areaList)			
				linkList=matchIndices(areaList,'Area')
				LL.append(linkList)		
			#Extract regionprops
			masterList.append(regionP)
			bwL0=bwL.copy()

		fileNum0=fileNum
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

	tr=removeShortTracks(tr,3)

	print "fixing loose ends"
	#Find possible link candidates, 1st pass
	mList=fixTracks(tr,2,[0,2])
	#Match the indices of the link candidates
	tr=reassignTrackID(tr,mList)
	
	print "fixing loose ends"
	
	#Find possible link candidates, 2nd pass
	mList=fixTracks(tr,2.5,[1,3])
	#Match the indices of the link candidates
	tr=reassignTrackID(tr,mList)
	print "fixing loose ends"
	
	#Find possible link candidates, 3rd pass
	mList=fixTracks(tr,3,[3,5])
	#Match the indices of the link candidates
	tr=reassignTrackID(tr,mList)
	
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
	print "Link mother-daughter tracks"
	tr=assignDaughter(trIn[:,0:8],3,[4,6])
	tr=assignDaughter(tr[:,0:8],3,[4,6])
	tr=assignDaughter(tr[:,0:8],3,[4,6])
	tr=assignDaughter(tr[:,0:8],3,[4,6])
	

	print "rename Tracks"
	#Remove orphaned cells and cells that do not divide
	tr=renameTracks(tr)	

	print "find family IDs"
	tr=findFamilyID(tr)
	
	print "match family IDs"
	tr=matchFamilies(tr)	
	tr=matchFamilies(tr)	

	print "fix family IDs"
	tr=fixFamilies(tr)	

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
	
	listOfMatches=matchTracks(masterEndList,masterStartList,
				  dist,timeRange)

	return listOfMatches	
	#Reassign the track IDs

def reassignTrackID(tr,matchList):
	'''
	tr=reassignTrackID(tr,matchList) use the output of fixTracks 
	to reassing the proper id to matched tracks
	'''

	shortTrack=3
	#Match the indices of the link candidates
	matchL=matchIndices(matchList.take([0,1,3],axis=1),'Distance')

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
	
		#Correct IDs in matchL
		matchL[matchL[:,0]==id1,0]=id0

        tr2=tr2[tr2[:,2].argsort(-1,'mergesort'),]
        tr2=tr2[tr2[:,3].argsort(-1,'mergesort'),]

	return tr2

def removeShortTracks(tr,shortTrack=3):
	'''
	tr=removeShortTracks(tr,shortTrack=3) removes 
	tracks shorter than shortTrack
	'''
	trLength=mp.pyplot.hist(tr[:,3],np.unique(tr[:,3]))
        idLong=trLength[1][trLength[0]>shortTrack]
        k=0
        #Reassign cell ID
        trS=np.zeros(tr[0,:].shape)
        for id in idLong:
                k=k+1
                trTemp=tr[tr[:,3]==id,:]
                trTemp[:,3]=k
		trS=np.vstack([trS,trTemp])
	trS=trS[trS[:,2].argsort(-1,'mergesort'),]
	trS=trS[trS[:,3].argsort(-1,'mergesort'),]
	return trS
	

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

		#Find starting points that fall inside the bounding box
		nearPts=inList2[(inList2[:,2]<=(pts[2]+timeRange[1])) & 
				(inList2[:,2]>=(pts[2]-timeRange[0])),:]
		inIndx=nx.points_inside_poly(nearPts[:,0:2],boxPts)		
		
		if inIndx.any():
			inPts=nearPts[inIndx,:]
			
			pole1=np.array([[pts[0] + pts[4]/2*np.cos(pts[6]*np.pi/180),
				         pts[1] + pts[5]/2*np.sin(pts[6]*np.pi/180)]])
			pole2=np.array([[pts[0] - pts[4]/2*np.cos(pts[6]*np.pi/180),
					 pts[1] - pts[5]/2*np.sin(pts[6]*np.pi/180)]])
			poleIn1=np.array([[inPts[:,0] + inPts[:,4]/2*np.cos(inPts[:,6]*np.pi/180),
				           inPts[:,1] + inPts[:,5]/2*np.sin(inPts[:,6]*np.pi/180)]])
			poleIn2=np.array([[inPts[:,0] - inPts[:,4]/2*np.cos(inPts[:,6]*np.pi/180),
				           inPts[:,1] - inPts[:,5]/2*np.sin(inPts[:,6]*np.pi/180)]])
	
				
			#Compute the distance between each one and the track	
			VL=eucledianDistances(pole1,poleIn1)
			VL=VL[0][0]
			VR=eucledianDistances(pole2,poleIn2)
			VR=VR[0][0]
			VC=eucledianDistances(np.array([pts[0:2]]),inPts[:,0:2])	
			VC=VC[0]

			T=5*np.hypot(0,pts[3]-inPts[:,3])
			#Put the possible candidates into an array		
			
			for id in range(len(VL)):
				V=np.array([VL[id],VR[id],VC[id]])
				Dist=np.array([[pts[3],inPts[id,3],
					     pts[2],T[id]+V.min()]]) 			
				matchList=np.append(matchList,Dist,0)
	#Match start and finishes with matchIndices
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
	
	arrayOut[0]=listIn[0].copy()
	
	print arrayOut

	k=0
	for id in range(1,len(listIn)):
		if np.size(arrayOut[k])==1:
			arrayOut[k]=listIn[id]
		elif len(listIn[id])>0:
			arrayOut[k]=np.vstack((arrayOut[k],listIn[id]))
		if np.mod(id,nID)==0:
			k=k+1

	arrayOutAll=arrayOut[0].copy()
	for id in range(1,len(arrayOut)):
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
			if len(tr)>25:
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
	win_size=10
	divJumpSize=15
	if len(L)>win_size:
		std_thres=0.15
		#Remove the points with a large positive derivative, do this twice
		jumpID=(np.diff(L)>divJumpSize).nonzero()[0]
		L[jumpID+1]=L[jumpID]
		
		jumpID=(np.diff(L)>divJumpSize).nonzero()[0]
		L[jumpID+1]=L[jumpID]
		
		Lw=rolling_window(L,win_size)
		Lstd=np.std(Lw,-1)/np.mean(Lw,-1)
		Li=(Lstd>std_thres)
		B=np.arange(len(Li))+win_size/2
		if Li[0]:
			Li[0]=False
		if Li[-1]:
			Li[-1]=False
		#Create list of points limiting the region of interest
		Blist=B[np.diff(Li)]
		
		
		#If the number delimiting point is odd
		if np.mod(len(Blist),2)==1:
			if Li[0]:
				Blist=np.append(np.array([0]),Blist)
			else:
				Blist=np.append(Blist,np.array([len(L)]))
		Blist=Blist.reshape(-1,2)				
		#Find the general location of a division event.
		divLoc=(np.diff(L)<-divJumpSize).nonzero()
		#Go through each point in the track
		for pt in Blist:
			E=divLoc[0][(divLoc[0]>pt[0])&(divLoc[0]<pt[1])]	
			if E.any():
				divTimes=np.append(divTimes,np.max(E))
		#Check if length is higher later
		for i in range(len(divTimes)):
			divs=divTimes[i]
			if L[divs:(divs+10)].max()>L[divs]:
				divTimes[i]=0
	divTimes=divTimes[divTimes!=0]
	divTimes=divTimes[divTimes>10]	
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

def removePeaks(LL):

	alist,blist=peakdet(LL,15)
	for pt in aList:
		Lmean=(LL[pt[0]-1]+LL[pt[0]+1])/2.
		Lplus=(LL[pt[0]]-LL[pt[0]-1])
		Lminus=(LL[pt[0]]-LL[pt[0]+1])

		print (abs(Lplus)-abs(Lminus))
		if (abs(Lplus)-abs(Lminus))<15:
			LL[pt[0]]=Lmean
	for pt in bList:
		Lmean=(LL[pt[0]-1]+LL[pt[0]+1])/2.
		Lplus=(LL[pt[0]]-LL[pt[0]-1])
		Lminus=(LL[pt[0]]-LL[pt[0]+1])

		print (abs(Lplus)-abs(Lminus))
		if (abs(Lplus)-abs(Lminus))<15:
			LL[pt[0]]=Lmean
	return LL
	


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)	

def findDD(trIn):
	divData=findDivisionEvents(trIn)
	return divData

def assignDaughter(trIn,dist=2,timeRange=[4,6]):
	'''

	'''
	divData=findDivisionEvents(trIn)	
		
	m=np.zeros((len(divData),8))
	k=0
	#Extract the track information at every division events
	for pt in divData:
		mt=trIn[(trIn[:,3]==pt[0])&(trIn[:,2]==pt[1]),:]
		m[k,:]=mt[0]
    		k=k+1		
	
	#Find the closest cells to every division events.	
	masterStartList=trIn[np.roll(np.diff(trIn[:,3])>0,1),:]
	
	#daughterMatchList=matchTracks(m,masterStartList,3)
	daughterMatchList=matchTracks(m,masterStartList,dist,timeRange)
	#Create an array containing [motherID, daughterID, time]	
	matchL=np.zeros((1,3))
	for t in np.unique(daughterMatchList[:,2]):
		mt=daughterMatchList[daughterMatchList[:,2]==t,:]
		if mt.any():
			ml=matchIndices(mt.take([0,1,3],axis=1),'Distance')		
			ml[:,2]=t
			matchL=np.append(matchL,ml,0)
			for id in ml[:,1]:
				daughterMatchList=daughterMatchList[daughterMatchList[:,1]!=id,:]
#	return matchL
	#Add a new column to trIn. Put in this column the ID of the daughter cell.
	#Also, put the ID of the mother	
#	print matchL
	trIn=np.hstack([trIn,np.zeros((len(trIn),1))])
	for k in reversed(range(len(matchL)-1)):
		div=matchL[k+1,:]
			
		trIn[(trIn[:,2]==div[2])&(trIn[:,3]==div[0]),8]=div[1]
		birthTime=trIn[trIn[:,3]==div[1],2]
		birthID=(trIn[:,3]==div[1]).nonzero()
		trIn[birthID[0][0],8]=-div[0]
		trIn=relabelCells(trIn,div)	
	trIn=trIn[trIn[:,2].argsort(-1,'mergesort'),]
	trIn=trIn[trIn[:,3].argsort(-1,'mergesort'),]
	
	return trIn

		
def relabelCells(tr,div):
	#Relabel mother and daughter cells so that the mother is always on the 'left'
	motherID=div[0]
	daughterID=div[1]
	divisionTime=div[2]
	motherDivLoc=tr[(tr[:,3]==motherID)&(tr[:,2]>(divisionTime+2)),0]		
	daughterDivLoc=tr[(tr[:,3]==daughterID)&(tr[:,2]>(divisionTime+2)),0]		

	if motherDivLoc.any()&daughterDivLoc.any():
		if motherDivLoc[0]>daughterDivLoc[0]:
			motherArr=(tr[:,3]==motherID)&(tr[:,2]>divisionTime)		
	
			daughterArr=(tr[:,3]==daughterID)&(tr[:,2]>divisionTime)

			tr[motherArr,3]=daughterID
			tr[daughterArr,3]=motherID
			
		daughterArrRem=(tr[:,3]==daughterID)&(tr[:,2]<=divisionTime)		
		if np.sum(daughterArrRem)>0:
			tr=tr[np.invert(daughterArrRem),:]
	

	return tr

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
	trOut=trOut[np.diff(trOut[:,2])>0,:]

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
				famList0=trI[int(id)][trI[int(id)][:,8]>0,8]
				famList=np.unique(np.hstack([famList,
					                     famList0]))
			if np.array_equal(famList,famList1):
				stop=True	
		for id in famList:
			trI[int(id)][:,9]=famID
		cellIdList=np.setdiff1d(cellIdList,famList)
		stop=False
		
	trLong=trI[0].copy()
	for id in range(1,len(trI)):
		if not np.isscalar(trI[id]):
			trLong=np.vstack([trLong,trI[id]])
	return trLong
		
def matchFamilies(trIn):
	'''
	This script matches the start of a new family with 
	the closest cell.	
	'''
	trIn=np.hstack([trIn,np.zeros((len(trIn),1))])	

		
	trFam=splitIntoList(trIn,9)
	trI=splitIntoList(trIn,3)	
	masterStartList=trIn[np.roll(np.diff(trIn[:,3])>0,1),:]
        divisionList=trIn[trIn[:,8]>0,:]	
	#find track ends
	
	
	dist=15	
	famMatchL=np.array([0,0,0,0,0]) #[oldFam, newFam, time, motherID, daughterID]
	
	# Loop over each family, find id of family closest to its start, 
	# store that in famMatchL=[newfamID, oldFamID]

	for famID in range(len(trFam)):
		if len(trFam[famID])>0:
			if trFam[famID][0,2]>10:
				listOfMatches=matchTracks([trFam[famID][0,:]],
							   trIn[10:,:],
							   dist,[50,-1])
				if listOfMatches.any():
					minD=listOfMatches[:,3].min()
					listOfMatches=listOfMatches[listOfMatches[:,3]<2*minD,:]
					for i in range(len(listOfMatches)):  #Make sure parent did not divide
						pt=listOfMatches[i,:]
						if trIn[(trIn[:,3]==pt[1]) & (trIn[:,2]==pt[2]),8]:
							listOfMatches[i,3]=10*minD					
						if len(trIn[(trIn[:,3]==pt[1]) & (trIn[:,2]==pt[2]),8])==0:
							listOfMatches[i,3]=10*minD					
					minD=listOfMatches[:,3].min()
					match=listOfMatches[(listOfMatches[:,3]==minD).nonzero(),:][0][0]
					famMatchL=np.vstack([famMatchL,np.array([famID,trI[int(match[1])][0,9],match[2],match[1],match[0]])])
					trIn[(trIn[:,3]==match[1])&(trIn[:,2]==match[2]),8]=match[0]
	
	#Loop through the [newFamID,oldFamID] array, updates the negative famID to the 10th column 
	for id in reversed(range(len(famMatchL))):
		pt=famMatchL[id]
		if pt.any():
			trI[int(pt[3])][trI[int(pt[3])][:,2]==pt[2],8]=pt[4]
			trI[int(pt[3])][trI[int(pt[3])][:,2]==pt[2],8]=pt[4]
			trI[int(pt[4])][0,8]=-pt[3]
        trOut=trI[0].copy()
        for id in range(1,len(trI)):
	        if not np.isscalar(trI[id]):
                        trOut=np.vstack([trOut,trI[id]])	
	trOut=findFamilyID(trOut[:,0:9])
	
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
	Adds the age of the cells in the last column
	'''	
	divData=findDivisionEvents(trIn)			
	
	trIn=np.hstack([trIn,np.zeros((len(trIn),1))])	

	trIn[np.roll(np.diff(trIn[:,3])>0,1),10]=1		

	for pt in divData:
		id=pt[0]
		time=pt[1]
		trIn[(trIn[:,3]==id)&(trIn[:,2]==time),10]=1			

	k=0
	for i in range(len(trIn)):
		if trIn[i,10]==1:
			k=0
		trIn[i,10]=k
		k=k+1
	
	return trIn	

def addElongationRate(trIn):
	'''
	Fits an exponential to the length between every divisions to find the elongation rate.
	'''
	divLocations=(trIn[:,10]==0).nonzero()[0]

	elongationRate=np.zeros((len(trIn),1))

	for loc in range(len(divLocations)-1):
		dataRange=range(divLocations[loc],divLocations[loc+1])	
		dT=trIn[dataRange,4]
		if len(dT)>20:
			dT=dT[5:-5]
			z = np.polyfit(range(len(dT)), np.log(dT), 1)
			elongationRate[dataRange]=z[0]	
	
	trIn=np.hstack([trIn,elongationRate])

	return trIn


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
		bwImg=processImage(img,scaleFact=1,sBlur=0.5,sAmount=0,
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
		LINKTRACKS=raw_input('Do you want to link the cell tracks? (yes or no) ')
		if LINKTRACKS=='yes':
			PROCESSTRACKS=raw_input('Do you want to analyze the cell tracks? (yes or no) ')
		SAVEPATH=(raw_input('Please enter the location where the analyzed files will be saved (leave empty to use current location) ') or './')
	else:
		FILEPATH='./'
		PROCESSFILES='yes'
		LINKTRACKS='yes'
		PROCESSTRACKS='yes'
		SAVEPATH='./'

	np.savez(SAVEPATH+'processFiles.npz',lnoise=lnoise,lobject=lobject,thres=thres)

	if PROCESSFILES=='yes':
		masterL,LL,AA=trackCells(FILEPATH,np.double(lnoise),np.double(lobject),np.double(thres))
		#np.save(SAVEPATH+'masterL.npy',masterL)			
		#np.save(SAVEPATH+'LL.npy',LL)			

	if LINKTRACKS=='yes':
		#masterL=np.load(SAVEPATH+'masterL.npy')
		#LL=np.load(SAVEPATH+'LL.npy')
		tr=linkTracks(masterL,LL)
		if PROCESSTRACKS=='yes':
			tr=processTracks(tr)
		with open(SAVEPATH+'/trData.dat', 'wb') as f:	
			f.write(b'# xPos yPos time cellID cellLength cellWidth cellAngle avgIntensity divisionEvents familyID cellAge elongationRate\n')
			np.savetxt(f,tr)
			print 'The analysis is complete. Data saved as '+SAVEPATH+'trData.dat'	
		
