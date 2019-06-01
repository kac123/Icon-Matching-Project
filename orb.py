def orb_score(img_kp1, img_kp2, matcher):
	if img_kp1 is None:
		return 0
	if img_kp2 is None:
		return 0
	matches = matcher.match(img_kp1,img_kp2)
	score=len(matches)
	return score

def query_orb_kp_database(img_kp, database):
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
	query_results = [orb_score(img_kp, db_kp, matcher) for db_kp in tqdm_notebook(database)]
	query_results = [(i,j) for i,j in enumerate(query_results)]
	sorted_results = query_results#.sort(reverse=True, key=lambda x:x[1])
	best = 1
	for i in sorted_results:
		if i[1] > best:
			best = i[1]
	normalized_results = [(x[0],round(100 * x[1]/best)) for x in sorted_results]
	return normalized_results

def run_orb_image(img, database):
	orb = cv2.ORB_create()
	#img = gray(img)    # queryImage
	# find the keypoints and descriptors with orb
	kp1, des1 = orb.detectAndCompute(img,None)

	if len(kp1) < 2:
		des1 = None
    
	return query_orb_kp_database(des1, database)