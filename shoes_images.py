import os
import h5py
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import cv2
import math
from tqdm import tqdm
from PIL import Image 
import mahotas 
from sklearn.preprocessing import normalize

# save and load pickle objects 
def save_obj(obj, name ):
	with open('obj/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		
def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)		

# create grayscale image from database image
def gray( img ):

	if np.shape(img)[2] == 3:
		array =  np.dot( img, [0.299, 0.587, 0.114] )
	else:
		array =  np.dot( np.transpose( img, (1,2,0)), [0.299, 0.587, 0.114] )
	
	return array.astype(np.uint8)

# find max distance given a list of points 
def find_max( point_list):

	max_dist = 0.000
	first_point = point_list[0]
	second_point = point_list[1]

	for i in range(0, len(point_list)-1):
		for j in range(i+1, len(point_list)):
			
			dist_sq = (point_list[i][0] - point_list[j][0])**2 + (point_list[i][1] - point_list[j][1])**2  
			
			if dist_sq > max_dist:
				max_dist = dist_sq
				first_point = point_list[i]
				second_point = point_list[j]
				
	return first_point, second_point, math.sqrt(max_dist)

# if object is a circle, find max distance which also has the closest slope to the best fit slope 
def find_tiebreaker( x, y, best_slope, max_dist ):

	first_point = (x[0], y[0])
	second_point = (x[1], y[1])
	min_slope_diff = 999999.999
	
	for i in range(0, len(x)-1):
		for j in range(i+1,len(x)+i):
		
			if j >= len(x):
				j = j % len(x)
			
			dist = math.sqrt( (x[i] - x[j])**2 + (y[i] - y[j])**2 )
			if (max_dist - dist)/max_dist <= 0.08 and (x[i] - x[j]) != 0 :
				slope_diff = abs( (y[i] - y[j])/(x[i] - x[j]) - best_slope )
				
				if slope_diff < min_slope_diff:
					min_slope_diff = slope_diff
					first_point = (x[i], y[i])
					second_point = (x[j], y[j])	

	return first_point, second_point
	
	
# find midpoints between two points given a set of fractions  
def get_midway (p1, p2, fractions):
	
	midpts = []
	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]
	
	for n, fract in enumerate(fractions):
		x = p1[0] + fract * dx 
		y = p1[1] + fract * dy 
		midpts.append( (x,y) )
			
	return midpts

# find intersection of 2 lines 	
def line_intersect(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 
	
	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]
	
	div = det(xdiff, ydiff)
	if div == 0:
		return (-999999,-999999) 

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y	
	
# find points of intersection between normal line and all possible line segments in contour 
def find_intersect( x1, y1, x2, y2):
	
	intersect = []
	
	for i in range(0,len(x2)):
		j = i + 1
		if j == len(x2):
			j = 0
		
		# find point of intersection
		point = line_intersect( ((x1[0], y1[0]), (x1[-1], y1[-1])), ((x2[i], y2[i]), (x2[j], y2[j])) )
		
		# check that the point lies on the contour 
		dotproduct = (point[0] - x2[i]) * (x2[j] - x2[i]) + (point[1] - y2[i])*(y2[j] - y2[i])
		squaredlength = (x2[j] - x2[i])**2 + (y2[j] - y2[i])**2
		
		if dotproduct >= 0 and dotproduct <= squaredlength and point not in intersect:
			intersect.append( point )

	return intersect
		
# create initial nested dictionary which will hold mid point and normalized distances to the contour 
fractions = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
w = 0
hash_obj_c = {}
hash_obj_z = {}
orig_path = os.getcwd()
path = 'C:\\Users\\kchad\\Documents\\Logo Project\\ut-zap50k-images-square\\ut-zap50k-images-square\\Shoes\\Sneakers and Athletic Shoes'
os.chdir(path)
items = os.listdir(".")

for item in tqdm(items):
	
	items2 = os.listdir(item)
	
	for item2 in items2:
		
		images = cv2.imread(os.path.join(path,item,item2))
		#plt.imshow(images)
		#plt.show()
		images = np.array(images)

		# create grayscale image and use Canny edge detection
		Data = gray(images)
		edges1 = cv2.Canny(Data,100,200)
		if np.amax(edges1,axis=(0,1)) == 0:
			edges1 = cv2.Canny(Data,20,100)
		if np.amax(edges1,axis=(0,1)) == 0:
			edges1 = cv2.Canny(Data,5,20)	
		if np.amax(edges1,axis=(0,1)) == 0:
			edges1 = cv2.Canny(Data,1,5)		
		
		rows,cols = Data.shape[:2]
		min_x = cols 
		min_y = rows
		max_x = 0 
		max_y = 0 
	
		# fill in any diagonals in the edges 
		for i in range(0, rows):
			for j in range(0, cols):
				if  i >= 1 and i <= rows - 2 and  j >= 1 and j <= cols - 2: 
					if edges1[i,j] == 255 and edges1[i,j-1] == 0 and edges1[i+1,j] == 0 and edges1[i+1,j-1] == 255:
						edges1[i+1,j] = 255 
					elif edges1[i,j] == 255 and edges1[i,j+1] == 0 and edges1[i+1,j] == 0 and edges1[i+1,j+1] == 255:
						edges1[i+1,j] = 255 
				
				if edges1[i,j] == 255:
					if j > max_x:
						max_x = j 
					if j < min_x:
						min_x = j
					if i > max_y:
						max_y = i 
					if i < min_y: 
						min_y = i
						
		edges2 = edges1[min_y:max_y+1, min_x:max_x+1]
		try:
			edges2 = cv2.resize(edges2, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
			zernike = mahotas.features.zernike_moments(edges2, 16)	
			zernike = normalize(zernike[:,np.newaxis], axis=0).ravel()	
	
		except:
			zernike = [0] * 25
	
		hash_obj_z[w] = zernike 
		# find contours from edge image, sort by length and take top 5 
		image, contours, hierarchy = cv2.findContours(edges1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key = lambda x:cv2.contourArea(x,False), reverse = True)[:10]
		
		# set min length of contour to 11
		n = 0
		while n < len(contours) and cv2.arcLength(contours[n],False) >= 11:
			n += 1
			
		contours = contours[:n]	
		
		dist_append = []
		
		# find hull and defects of the contour 
		for n in range(0,len(contours)):
		
			cnt = contours[n]
			hull = cv2.convexHull(cnt,returnPoints = False)
			defects = cv2.convexityDefects(cnt,hull)
			
			# initialize list to track all points on the hull as well as points with defects 
			point_track = []
			
			for j in range (hull.shape[0]):	
				i = hull[j]
				point = tuple(cnt[i][0][0])
				point_track.append(point)
			
			if defects is not None:
				for i in range(defects.shape[0]):
					s,e,f,d = defects[i,0]
					far = tuple(cnt[f][0])
					point_track.append(far)
			
			# 4b. find max distance and plot 2 points along with longest line 
			if len(point_track) >= 2:
				[point1, point2, max_dist] = find_max(point_track)
				
				# find area of contour and enclosing circle
				area = cv2.contourArea(cnt)
				(cx, cy), radius = cv2.minEnclosingCircle(cnt)
				circleArea = round(radius**2 * np.pi,1)
				# if contour is a circle, then find best fit line, then find points which have a max distance and the same slope
				if abs(circleArea - area)/circleArea <= .25:
					coordinates = np.where(edges1 == 255)
					if len(coordinates[1]) > 0:
						coef = np.polyfit(coordinates[1], coordinates[0], 1, rcond=None, full=False, w=None, cov=False)
						[point1, point2] = find_tiebreaker(cnt[:,0,0], cnt[:,0,1], coef[0], max_dist)
			
				# find midway points along line segment  
				midpts = get_midway( point1, point2, fractions ) 
				
				# slope of normal line 
				if (point2[0]-point1[0]) == 0:
					slope = 0
				elif (point2[1]-point1[1]) == 0:
					slope = 10000
				else:
					slope = -1 / ( (point2[1]-point1[1]) / (point2[0]-point1[0]) )
			
				# 4d. loop through the middle/quarter/eighth points and plot the normal lines 
				for m, pt in enumerate(midpts):
		
					x = np.linspace(0.0,cols,num=3)
					y = [float(slope * i - slope * pt[0] + pt[1]) for i in x]  
					
					if n == 0:
						for n1 in range(0,len(contours)):
							cnt1 = contours[n1]	
						
							# find points of intersection between normal line and contour, plot those points
							intersect = find_intersect(x, y, cnt1[:,0,0], cnt1[:,0,1])
							
							# distance between middle point and intersect points, normalize using length of longest line
							dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / max_dist , 3 ) for i in intersect ]
							for d in dist:
								if (fractions[m],d) not in dist_append:
									dist_append.append((fractions[m],d))
									
					else:
						# find points of intersection between normal line and contour, plot those points
						intersect = find_intersect(x, y, cnt[:,0,0], cnt[:,0,1])
						
						# distance between middle point and intersect points, normalize using length of longest line
						dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / max_dist , 3 ) for i in intersect ]
						for d in dist:
							if (fractions[m],d) not in dist_append:
								dist_append.append((fractions[m],d))					
						
		
		for d in dist_append:
			hash_obj_c.setdefault(d[0],[]).append((d[1],w))
			
		w = w + 1

os.chdir(orig_path)	
save_obj(  hash_obj_c, 'Contour_Database_Shoes' )
save_obj(  hash_obj_z, 'Zernike_Database_Shoes' )
