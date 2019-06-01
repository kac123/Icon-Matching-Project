import sys 
import os
import h5py
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import cv2
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm, tnrange, tqdm_notebook
import collections
import random
import mahotas 
from sklearn.preprocessing import normalize
import logging
from glob import iglob
from util import save_obj

from aberrations import ab_choice
from plotter import plot_results, log_results
import contour
import zernike
import combined
import sift
import orb	
	

def load_images():
	return [i for i in map(cv2.imread, iglob('shoes/**/*.jpg', recursive=True)) if i is not None]
# simple way to load the complete dataset (for a more sophisticated generator example, see LLD-logo script)
# open hdf5 file
# load data into memory as numpy array
#images, labels = (hdf5_file['data'][:], hdf5_file['labels/resnet/rc_64'][:])

# alternatively, h5py objects can be used like numpy arrays without loading the whole dataset into memory:
images = load_images()
# here, images[0] will be again returned as a numpy array and can eg. be viewed with matplotlib using

# load the dictionaries with full database results 
# fraction1: (dist1, logo ID), (dist2, logoID), ...
print("Loading Orb Database")
hash_orb = load_obj("orb_database")
print(len(hash_orb))
print("Loading Contour Dictionary")
hash_obj = load_obj('Contour_Database_Shoes')
print(len(hash_obj))
print("Loading Zernike Vectors")
hash_zernike = load_obj('Zernike_Database_Shoes')
print(len(hash_zernike))
print("Loading Sift Database")
hash_sift = load_obj("sift_database")
print(len(hash_sift))

print("Loading Completed")



#####################
# main loop 	#####
##################### 0 1 2 3 7 8

# load image  and apply random mutation 
avg_rankings = [[0,0,0,0,0] for i in range(12)]
correct_top5 = [[0,0,0,0,0] for i in range(12)]
correct_top10 = [[0,0,0,0,0] for i in range(12)]
image_index = -1
for img in images:
	if img is None:
		continue
	image_index+=1
	Data = gray(img)
	for ab_index in range(12):
		Data, mutation = ab_choice(Data, n = ab_index)
		print (mutation)

		# define fractions along line segment to be used 
		fractions = [.1,.2,.3,.4,.5,.6,.7,.8,.9]

		# create queries from the image 
		contour_query = contour.create_query( Data, fractions)
		zernike_query = zernike.create_query( Data, fractions)

		# obtain results from ORB  
		orb_list = run_orb_image(Data, hash_orb)
		print("Orb score: ",orb_list[image_index]) 

		# obtain results from SIFT  
		sift_list = run_sift_image(Data, hash_sift)
		print("Sift score: ",sift_list[image_index]) 

		# obtain results from contour method and zernike method 
		zernike_list = zerike.test_query(zernike_query)	
		print("Zernike score: ",zernike_list[image_index])
		contour_list = contour.test_query( contour_query, 0.002, fractions )
		print("Contour score: ",contour_list[image_index])

		# combine scores of each method and log the results, input the weights along with the results from each method  
		weights = [.25,.25,.25,.25]
		matched_list_v2 = test_combined(weights, contour_list, zernike_list, sift_list, orb_list)
		orb_list = sorted(orb_list, key = lambda tup: tup[1], reverse = True )
		sift_list = sorted(sift_list, key = lambda tup: tup[1], reverse = True )	
		zernike_list = sorted(zernike_list, key = lambda tup: tup[1], reverse = True )	
		contour_list = sorted(contour_list, key = lambda tup: tup[1], reverse = True )	


		# find the rank of the original image, just for checking purposes 	
		rank = 0
		for hit in contour_list:
			rank += 1
			if hit[rank] == image_index:
				avg_rankings[ab_index][0] += rank
				if rank <= 5:
					correct_top5[ab_index][0] += 1
				if rank <= 10:
					correct_top10[ab_index][0] += 1
				break
		rank = 0
		for hit in zernike_list:
			rank += 1
			if hit[rank] == image_index:
				avg_rankings[ab_index][1] += rank
				if rank <= 5:
					correct_top5[ab_index][1] += 1
				if rank <= 10:
					correct_top10[ab_index][1] += 1
				break
		rank = 0
		for hit in sift_list:
			rank += 1
			if hit[rank] == image_index:
				avg_rankings[ab_index][2] += rank
				if rank <= 5:
					correct_top5[ab_index][2] += 1
				if rank <= 10:
					correct_top10[ab_index][2] += 1
				break
		rank = 0
		for hit in orb_list:
			rank += 1
			if hit[rank] == image_index:
				avg_rankings[ab_index][3] += rank
				if rank <= 5:
					correct_top5[ab_index][3] += 1
				if rank <= 10:
					correct_top10[ab_index][3] += 1
				break
		rank = 0
		for hit in matched_list_v2:
			rank += 1
			if hit[rank] == image_index:
				avg_rankings[ab_index][4] += rank
				if rank <= 5:
					correct_top5[ab_index][4] += 1
				if rank <= 10:
					correct_top10[ab_index][4] += 1
				print("Final ranking: ",rank,hit)
				break
		
	# create a log file of the results 
	log_results ( image_index, mutation, matched_list_v2, contour_list, zernike_list, sift_list, orb_list, avg_rankings, correct_top5, correct_top10, image_index+1 )