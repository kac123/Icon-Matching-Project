import cv2
import numpy as np
import math

from util import image_preprocess, find_max, find_tiebreaker, get_midway, find_intersect

#fuck if I know how this shit works, need to refactor to look like zernike.py

def create_query(img, fractions=[.1,.2,.3,.4,.5,.6,.7,.8,.9]):
	query_obj = {}
	rows, cols, _, _, _, _, edges1 = image_preprocess(img)	
	
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
	
				x = np.linspace(0.0,cols,3)
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

	
	## create final dictionary for contour method distances 
	for d in dist_append:
		query_obj.setdefault(d[0],[]).append(d[1])	
	
	return query_obj

def compare_queries(x,y , error = 0.002, fractions = [.1,.2,.3,.4,.5,.6,.7,.8,.9]) :
	# track all of the matched images
	total_points = 0
	matched_points = 0

	# loop through each of the fractions (key of the dictionary)
	for f in fractions:
	
		# loop through each distance in the database and check if the distance matches with the image query
		for hash_dist in  y.get(f) :
	
			total_points += 1
			
			# loop through each distance in the image query
			for query_dist in x.get(f):
		
				if query_dist >= hash_dist[0] - error and query_dist <= hash_dist[0] + error:
					matched_points += 1
	
	# count the number of matches for each image and find the top hits 	
	match = round(float(matched_points)*100/total_points,1) 
	return match 

def test_query(z, hash_obj):
	contour_list = []
	for img_index in hash_obj:
		contour_list.append((img_index, compare_queries(z, hash_obj[img_index])))
	return contour_list	


def generate_database(images):
	contour_database  = {}
	for img_index in range(len(images)):
		img = images[img_index]
		if img is not None:
			contour_database[img_index] = create_query(img)
	return contour_database