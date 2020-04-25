from util import gray
import cv2

def create_query(img):
	sift = cv2.xfeatures2d.SIFT_create()
	img = cv2.resize(img,None, fx=13, fy=13, interpolation=cv2.INTER_AREA)
	img = gray(img)
	# find the keypoints and descriptors with sift
	kp1, des1 = sift.detectAndCompute(img,None)
	if len(kp1) < 2:
		des1 = None
	return des1

def compare_queries(img_kp1, img_kp2):
	if img_kp1 is None:
		return 0
	if img_kp2 is None:
		return 0

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	matcher = cv2.FlannBasedMatcher(index_params,search_params)

	try:
		matches = matcher.knnMatch(img_kp1, img_kp2, k=2)
	except:
		return 0
		
	score = 0
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			score += 1
	return score

def test_query(img_kp, database):
	sift_list = []
	for img_index in database:
		sift_list.append((img_index, compare_queries(img_kp, database[img_index])))
	best = 1
	for i in sift_list:
		if i[1] > best:
			best = i[1]
	sift_list = [(x[0],round(100 * x[1]/best)) for x in sift_list]
	return sift_list

def generate_database(images):
	sift_database  = {}
	for img_index in range(len(images)):
		img = images[img_index]
		sift_database[img_index] = create_query(img)
	return sift_database