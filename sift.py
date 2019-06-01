def sift_score(img_kp1, img_kp2, matcher):
		
	if img_kp1 is None:
		return 0
	if img_kp2 is None:
		return 0
	
	try:
		matches = matcher.knnMatch(img_kp1, img_kp2, k=2)
	except:
		return 0
		
	score = 0
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			score += 1
	return score

def query_sift_kp_database(img_kp, database):
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	matcher = cv2.FlannBasedMatcher(index_params,search_params)
	
	query_results = [sift_score(img_kp, db_kp, matcher) for db_kp in tqdm_notebook(database)]
	query_results = [(i,j) for i,j in enumerate(query_results)]
	sorted_results = query_results#.sort(reverse=True, key=lambda x:x[1])
	best = 1
	for i in sorted_results:
		if i[1] > best:
			best = i[1]
	normalized_results = [(x[0],round(100 * x[1]/best,1)) for x in sorted_results]
	return normalized_results

def run_sift_image(img, database):
	sift = cv2.xfeatures2d.SIFT_create()
	#img = gray(img) # queryImage
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img,None)

	if len(kp1) < 2:
		des1 = None
	
	return query_sift_kp_database(des1, database)