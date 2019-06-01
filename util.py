import h5py
import cv2
import numpy as np
import pickle
import math

RESOURCE_PATH = "res/"
# save and load pickle objects 
def save_obj(obj, name ):
    with open(RESOURCE_PATH + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		
def load_obj(name ):
    with open(RESOURCE_PATH + name + '.pkl', 'rb') as f:
        return pickle.load(f)		

# create grayscale image from database image
def gray( img ):
	img = img.astype('uint8')
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')

def load_images(filename = None):
	if filename: #LLD-icon.hdf5
		hdf5_file = h5py.File(RESOURCE_PATH + filename, 'r')
		images, labels = (hdf5_file['data'], hdf5_file['labels/resnet/rc_64'])
		return images
	return [i for i in map(cv2.imread, iglob('shoes/**/*.jpg', recursive=True)) if i is not None]

# find max distance given a list of points 
def find_max( point_list):

	max_dist = 0.000
	first_point = point_list[0]
	second_point = point_list[1]

	for i in range(0, len(point_list)-1):
		for j in range(i+1, len(point_list)):
			
			dist_sq =  (point_list[i][0] - point_list[j][0])**2 + (point_list[i][1] - point_list[j][1])**2 
			
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

def canny(img):
	cimg = cv2.Canny(img,100,200)
	if np.amax(cimg,axis=(0,1)) == 0:
		cimg = cv2.Canny(img,20,100)
	if np.amax(cimg,axis=(0,1)) == 0:
		cimg = cv2.Canny(img,5,20)
	if np.amax(cimg,axis=(0,1)) == 0:
		cimg = cv2.Canny(img,1,5)	
	return cimg

def fill_in_diagonals(img):
	rows,cols = img.shape[:2]
	min_x = cols 
	min_y = rows
	max_x = 0 
	max_y = 0 
	# fill in any diagonals in the edges 
	for i in range(0, rows):
		for j in range(0, cols):
			if  i >= 1 and i <= rows - 2 and j >= 1 and j <= cols - 2: 
				if img[i,j] == 255 and img[i,j-1] == 0 and img[i+1,j] == 0 and img[i+1,j-1] == 255:
					img[i+1,j] = 255 
				elif img[i,j] == 255 and img[i,j+1] == 0 and img[i+1,j] == 0 and img[i+1,j+1] == 255:
					img[i+1,j] = 255 
			
			if img[i,j] == 255:
				if j > max_x:
					max_x = j 
				if j < min_x:
					min_x = j
				if i > max_y:
					max_y = i 
				if i < min_y: 
					min_y = i
	return img, rows, cols, min_x, min_y, max_x, max_y

def image_preprocess(img):
	rows,cols = img.shape[:2]
	# create grayscale image and use Canny edge detection
	cimg = canny(img)	
	
	dcimg, rows, cols, min_x, min_y, max_x, max_y = fill_in_diagonals(cimg)
	
	return rows, cols, min_x, min_y, max_x, max_y, dcimg