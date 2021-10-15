import numpy as np
import cv2

def maybetocolor(img):
    # turn a 1 channel grayscale image into a 3 channel image
    # still gray, but now can be used for methods that normaly work on
    # color images
    if len(img.shape) != 3:
        return np.repeat(img[...,np.newaxis],3,axis=2)
    return img

def maybetogray(img):
    # turn a color image to a grayscale one
    # if the image is already grayscale, just return
    if len(img.shape) == 3:
        return np.dot(img, [0.299, 0.587, 0.114]).astype(np.uint8)
    return img

def add_border(img, size):
    rows,cols = img.shape[:2]
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    try:
        common = int(max(set(border), key = border.count))
        array = cv2.copyMakeBorder( img,  size, size, size, size, cv2.BORDER_CONSTANT, value =  common)
    except:
        common = sstats.mode(border)[0][0] # get the most common color
        common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
        array = cv2.copyMakeBorder( img,  size, size, size, size, cv2.BORDER_CONSTANT, value =  common)

    return array.astype(np.uint8)
    
def image_preprocess(img):
    img = maybetogray(img)
    rows,cols = img.shape[:2]
    # create grayscale image and use Canny edge detection
    cimg = canny(img)   
    
    dcimg, rows, cols, min_x, min_y, max_x, max_y = fill_in_diagonals(cimg)
    
    return rows, cols, min_x, min_y, max_x, max_y, dcimg    

def prep_img(img,mgray=True):
    if mgray:
        img = maybetogray(img)
    img = add_border(img, 16)
    rows,cols = img.shape[:2]
    scale = (128 / max(rows,cols))
    img = cv2.resize(img,  (int(rows * scale), int(cols * scale)))    
    return img

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def canny(img,**kwargs):
    # we try several threshold values and find the one with the most pixels
    #TODO# add some more methods to determine the threshold automatically, see which is best
    threshhold = [20,60,120, 200]
    t_max = np.argmax([
        np.sum(np.array(cv2.Canny(img,threshhold[t],threshhold[t+1])) >= 200)
                       for t in range(0,len(threshhold)-1)])
    edges1 = cv2.Canny(img,threshhold[t_max],threshhold[t_max+1])    
    return edges1

def fill_in_diagonals(img):
    rows,cols = img.shape[:2]
    min_x = cols 
    min_y = rows
    max_x = 0 
    max_y = 0 
    # fill in any diagonals in the edges 
    for i in range(0, rows):
        for j in range(0, cols):
            if  i >= 1 and i <= rows - 2 and j >= 1 and j <= cols - 2: 
                if img[i,j] == 255 and img[i,j-1] == 0 and img[i+1,j] == 0 and img[i+1,j-1] == 255:
                    img[i+1,j] = 255 
                elif img[i,j] == 255 and img[i,j+1] == 0 and img[i+1,j] == 0 and img[i+1,j+1] == 255:
                    img[i+1,j] = 255 
            
            if img[i,j] == 255:
                if j > max_x:
                    max_x = j 
                if j < min_x:
                    min_x = j
                if i > max_y:
                    max_y = i 
                if i < min_y: 
                    min_y = i
    return img, rows, cols, min_x, min_y, max_x, max_y