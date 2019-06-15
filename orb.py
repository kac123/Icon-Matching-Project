from util import gray
import cv2

def create_query(img):
	orb = cv2.ORB_create()
	img = cv2.resize(img,None, fx=13, fy=13, interpolation=cv2.INTER_AREA)
	img = gray(img)
	# find the keypoints and descriptors with orb
	kp1, des1 = orb.detectAndCompute(img,None)
	if len(kp1) < 2:
		des1 = None
	return des1

def compare_queries(img_kp1, img_kp2):
	if img_kp1 is None:
		return 0
	if img_kp2 is None:
		return 0
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = matcher.match(img_kp1,img_kp2)
	score=len(matches)
	return score

def test_query(img_kp, database):
	orb_list = []
	for img_index in database:
		orb_list.append((img_index, compare_queries(img_kp, database[img_index])))
	best = 1
	for i in orb_list:
		if i[1] > best:
			best = i[1]
	orb_list = [(x[0],round(100 * x[1]/best)) for x in orb_list]
	return orb_list

def generate_database(images):
	orb_database  = {}
	for img_index in range(len(images)):
		img = images[img_index]
		orb_database[img_index] = create_query(img)
	return orb_database