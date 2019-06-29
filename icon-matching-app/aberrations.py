def ab_translate(img):
	rows,cols = img.shape
	M = np.float32([[1,0,-6],[0,1,6]])
	img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
	return img

def ab_rotate(img):
	rows,cols = img.shape
	M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),random.randint(0,360),1)
	img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
	return img

def ab_affine(img):
	# cv2.getAffineTransform takes 3 input points and 3 output points, and returns the affine transformation matrix mapping the input points to the output.
	rows,cols = img.shape
	pts1 = np.float32([[0,0],[0,30],[30,30]])
	pts2 = np.float32([[5,5],[6,20],[22,21]])
	M = cv2.getAffineTransform(pts1,pts2)
	img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
	return img
    
def ab_scale(img):
	rows,cols = img.shape
	M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,random.uniform(.8, 1.25))
	img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
	return img

def ab_flip(img):
	img = cv2.flip(img, random.randint(0,1))
	return img

def ab_line(img):
	cv2.line(img,(16,20),(21,19),(0,300,300),2)
	cv2.line(img,(23,22),(18,20),(0,300,300),2)
	return img

def ab_circle(img):
	cv2.circle(img, (14,18), 3, (0,300,300), 2)
	cv2.circle(img, (19,13), 3, (0,300,300), 3)
	return img

def ab_line_circle(img):
	cv2.line(img,(11,10),(15,8),(0,300,300),2)        
	cv2.circle(img, (14,18), 3, (0,300,300), 2)
	return img   

def ab_two_line_circle(img):
	cv2.line(img,(6,20),(6,12),(0,300,300),2)  
	cv2.line(img,(10,25),(18,25),(0,300,300),2) 
	cv2.circle(img, (14,18), 3, (0,300,300), 2)
	cv2.circle(img, (19,12), 3, (0,300,300), 2)        
	return img 

def ab_trans_rot(img):
	return(ab_translate(ab_rotate(img)))

def ab_affine_rot(img):
	return(ab_affine(ab_rotate(img)))

def ab_scale_lc(img):
	return(ab_scale(ab_two_line_circle(img)))

aberrations = [ab_translate, ab_rotate, ab_affine, ab_scale, ab_flip, ab_line, ab_circle, ab_line_circle, ab_two_line_circle, ab_trans_rot, ab_affine_rot, ab_scale_lc]

def ab_choice(img, n=1):
	# This function takes an input image (32 by 32) and applies a random aberration to it, either some affine transformation or an occlusion N times.
	# Note that the input image is mutated.

	# Choose the given number of aberrations from the array to apply.
	ab = aberrations[n]
	img = ab(img)
	return img, ab