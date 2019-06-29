import cv2
import mahotas
from sklearn.preprocessing import normalize
import numpy as np

from util import image_preprocess

def create_query(img):
	_, _, min_x, min_y, max_x, max_y, edges1 = image_preprocess(img) 
	
	## create zernike vector 
	edges2 = edges1[min_y:max_y+1, min_x:max_x+1]
	edges2 = cv2.resize(edges2, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
	zernike = mahotas.features.zernike_moments(edges2, 16)	
	return normalize(zernike[:,np.newaxis], axis=0).ravel()

def compare_queries(x,y):
	dot_prod = sum(i[0] * i[1] for i in zip(x, y))
	return round(666.667 * (dot_prod - .850),1) 

def test_query(z, hash_zernike):
	zernike_list = []
	for img_index in hash_zernike:
		zernike_list.append((img_index, compare_queries(z, hash_zernike[img_index])))
	return zernike_list	

def generate_database(images):
	zernike_database  = {}
	for img_index in range(len(images)):
		img = images[img_index]
		zernike_database[img_index] = create_query(img)
	return zernike_database