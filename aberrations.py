def ab_id(img):
    return img

def ab_rotate_border(img):
    rows,cols,_ = img.shape
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    common = sstats.mode(border)[0][0] # get the most common color
    common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),random.randint(90,180),1)
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=common)
    return img

def ab_scale_border(img):
    rows,cols,_ = img.shape
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    common = sstats.mode(border)[0][0] # get the most common color
    common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,1.2)
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=common)
    return img

def ab_shrink_border(img):
    rows,cols,_ = img.shape
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    common = sstats.mode(border)[0][0] # get the most common color
    common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,0.8)
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=common)
    return img

def ab_translate(img):
    rows,cols,_ = img.shape
    M = np.float32([[1,0,-6],[0,1,6]])
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_rotate(img):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),random.randint(0,360),1)
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_affine(img):
    # cv2.getAffineTransform takes 3 input points and 3 output points, and returns the affine transformation matrix mapping the input points to the output.
    rows,cols,_ = img.shape
    pts1 = np.float32([[0,0],[0,30],[30,30]])
    pts2 = np.float32([[5,5],[6,20],[22,21]])
    M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_scale(img):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,random.uniform(.8, 1.25))
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_flip(img):
    img = cv2.flip(img, random.randint(0,1))
    return img

def ab_line(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.line(img,(16,20),(21,19),(0,300,300),2)
    cv2.line(img,(23,22),(18,20),(0,300,300),2)
    return img

def ab_circle(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.circle(img, (14,18), 3, (0,300,300), 2)
    cv2.circle(img, (19,13), 3, (0,300,300), 3)
    return img

def ab_line_circle(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.line(img,(11,10),(15,8),(0,300,300),2)        
    cv2.circle(img, (14,18), 3, (0,300,300), 2)
    return img   

def ab_two_line_circle(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
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

aberrations = [ab_id, ab_rotate_border, ab_scale_border, ab_shrink_border, ab_translate, ab_rotate,ab_affine,ab_scale,ab_flip,ab_line,ab_circle, ab_line_circle,ab_two_line_circle
                  ,ab_trans_rot,ab_affine_rot, ab_scale_lc]

# aberrations function
def get_ab(img, n=None):
    # This function takes an input image (32 by 32) and applies a random aberration to it, either some affine transformation or an occlusion N times.
    # Note that the input image is mutated.
    ab = None
    if n is not None and n >= 0 and n < len(aberrations): # if we want a specific ab then give it
        ab = aberrations[n]
    else:
        ab = np.random.choice(aberrations) # otherwise pick at random
    img = ab(img)
    return img, ab