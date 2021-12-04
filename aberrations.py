import numpy as np
import cv2
from scipy import stats as sstats
import random
from imageprocessing import bg_stats

def ab_id(img):
    return img

def translate_stream(img):
    yield img
    rows,cols,_ = img.shape
    for dx in range(-rows//2, rows//2):
        for dy in range(-cols//2, cols//2):
            M = np.float32([[1,0,dx],[0,1,dy]])
            yield cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_WRAP), (dx, dy)
            
def rotate_stream(img, border=None, borderMode=cv2.BORDER_REPLICATE):
    p_img = img
    yield p_img
    rows,cols,_ = img.shape
    for i in range(0, 360):
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),i,1)
        if border is not None:
            p_img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
        else:
            p_img = cv2.warpAffine(img,M,(cols,rows),borderMode=borderMode)
        yield p_img, i

        
def scale_stream(img, border=None, borderMode=cv2.BORDER_REPLICATE):
    p_img = img
    yield p_img
    rows,cols,_ = img.shape
    for i in range(60, 201):
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,i/100)
        if border is not None:
            p_img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
        else:
            p_img = cv2.warpAffine(img,M,(cols,rows),borderMode=borderMode)
        yield p_img, i

             

def stream(img, transformation, N, **kwargs):
    yield img
    for i in range(N):
        yield transformation(img, **kwargs)
        
def invariance_measure(method, stream):
    img_v = method.create_query(next(stream))
    for p_img, d in stream:
        yield (method.compare_queries(img_v, method.create_query(p_img)), d)


# the first 5 are the basic geometric transformations
def ab_translate(img, border=None, return_coords=False, limiter=0):
    rows,cols,_ = img.shape
    if limiter == 0:
        dx = random.randint(-rows,rows)
        dy = random.randint(-cols,cols)
    else:
        dx = random.randint(-limiter,limiter)
        dy = random.randint(-limiter,limiter)
    M = np.float32([[1,0,dx],[0,1,dy]])
    if border is not None:
        if border == "wrap":
            img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_WRAP)
        else:
            img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    if return_coords:
        return img, (dx,dy)
    return img

def ab_translate_safe(img, limit=1/5):
    height,width = img.shape[:2]
    ylim = int(height*limit)
    xlim = int(width*limit)
    center, (xmin,xmax,ymin,ymax) = bg_stats(img)
    xd = min(xlim, (xmax-xmin)/2)
    yd = min(ylim, (ymax-ymin)/2)
    dx = random.randint(-xd,xd)
    dy = random.randint(-yd,yd)
    M = np.float32([[1,0,dx],[0,1,dy]])
    nimg = cv2.warpAffine(img,M,(width,height),borderMode=cv2.BORDER_REPLICATE)
    return nimg

def ab_rotate(img, border=None, borderMode=cv2.BORDER_REPLICATE):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),random.randint(0,360),1)
    if border is not None:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=borderMode)
    return img

def ab_rotate_safe(img, angle=None):
    angle = angle or random.randint(0,359)
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rmat = cv2.getRotationMatrix2D(center, angle, 1.)
    abcos = abs(rmat[0,0])
    absin = abs(rmat[0,1])
    bound_w = int(height*absin + width*abcos)
    bound_h = int(height*abcos + width*absin)
    rmat[0,2] += bound_w/2 - center[0]
    rmat[1,2] += bound_h/2 - center[1]
    nimg = cv2.warpAffine(img,rmat,(bound_w,bound_h))
    return nimg

def ab_affine(img, border=None):
    # cv2.getAffineTransform takes 3 input points and 3 output points, and returns the affine transformation matrix mapping the input points to the output.
    rows,cols,_ = img.shape
    pts1 = np.float32([[0,0],[0,30],[30,30]])
    pts2 = np.float32([[random.randint(0,10),random.randint(0,10)],[random.randint(0,10),random.randint(20,30)],[random.randint(20,30),random.randint(20,30)]])
    M = cv2.getAffineTransform(pts1,pts2)
    if border is not None:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_scale(img, border=None):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,random.uniform(.8, 1.25))
    if border is not None:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_flip(img):
    img = cv2.flip(img, random.randint(0,1))
    return img

basic_geometric = [ab_translate, ab_rotate, ab_affine, ab_scale, ab_flip]

# the next two are the basic drawing transformations
def ab_line(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.line(img,(random.randint(15,25),random.randint(15,25)),(random.randint(15,25),random.randint(15,25)),(0,300,300),random.randint(1,3))
    return img

def ab_circle(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.circle(img, (random.randint(12,20),random.randint(12,20)), random.randint(2,4), (0,300,300), random.randint(-3, 3))
    return img

basic_drawing = [ab_line, ab_circle]

# border_color gets the most common color along the border to use when applying a geometric transformation
def border_color(img):
    rows,cols,_ = img.shape
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    common = sstats.mode(border)[0][0] # get the most common color
    common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
    return common

def ab_rotate_border(img):
    border = border_color(img)
    img = ab_rotate(img, border)
    return img

def ab_translate_border(img):
    border = border_color(img)
    img = ab_translate(img, border)
    return img

def ab_scale_border(img):
    border = border_color(img)
    img = ab_scale(img, border)
    return img

def ab_affine_border(img):
    border = border_color(img)
    img = ab_affine(img, border)
    return img

border_geometric = [ab_rotate_border,  ab_rotate_safe, ab_translate_border, ab_translate_safe, ab_scale_border, ab_affine_border]
all_geometric = border_geometric+[ab_flip]

# this applies multiple drawings
def ab_draw(img):
    num_drawings = random.randint(1,3)
    for i in range(num_drawings):
        img = np.random.choice(basic_drawing)(img)
    return img

# this applies multiple geometric transforms
def ab_warp(img):
    num_transforms = random.randint(1,3)
    for i in range(num_transforms):
        img = np.random.choice(all_geometric)(img)
    return img

# and this applies both
def ab_draw_warp(img):
    img = ab_draw(img)
    img = ab_warp(img)
    return img

all_advanced = [ab_draw, ab_warp, ab_draw_warp]

aberrations = [ab_id]+basic_drawing+all_geometric+all_advanced

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