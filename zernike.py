from util import image_preprocess

def create_query(img):
	_, _, min_x, min_y, max_x, max_y, edges1 = image_preprocess(img) 
	
	## create zernike vector 
	edges2 = edges1[min_y:max_y+1, min_x:max_x+1]
	edges2 = cv2.resize(edges2, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
	zernike = mahotas.features.zernike_moments(edges2, 16)	
	return normalize(zernike[:,np.newaxis], axis=0).ravel()

def test_query(z, images, hash_zernike):

	zernike_list = []
	for w in tqdm_notebook( range(len(images)) ):
		dot_prod = sum(i[0] * i[1] for i in zip(hash_zernike[w], z ))
		zernike_list.append(( w,round(666.667 * (dot_prod - .850),1) ))
	return zernike_list	