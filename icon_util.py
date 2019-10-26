#imports needed for the utility functions
import numpy as np
import pandas as pd
import cv2
import math
import pickle
from time import perf_counter


# the utility functions
 
def save_obj(obj, name ):
    # save pickle objects
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    # load pickle objects
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)        

def gray( img ):
    # create grayscale image from database image
    
    # if the image is already grayscale, just return
    if(len(np.shape(img)) < 3):
        return img
    
    if np.shape(img)[2] == 3:
        img = np.dot( img, [0.299, 0.587, 0.114] )
    else:
        img = np.dot( np.transpose( img, (1,2,0)), [0.299, 0.587, 0.114] )
    return img.astype(np.uint8)

def add_border(img):
    rows,cols = img.shape
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    common = int(max(set(border), key = border.count))
    array = cv2.copyMakeBorder( img,  10, 10, 10, 10, cv2.BORDER_CONSTANT, value =  common)
    return array.astype(np.uint8)

def prep_img(img):
    img = gray(img)
    img = add_border(img)
    return img

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


## all the functions related to zernike and contour
def find_max( point_list):

    max_dist = 0.000
    second_dist = 0.000
    first_point = point_list[0]
    second_point = point_list[1]
    third_point = point_list[0]
    fourth_point = point_list[1]

    for i in range(0, len(point_list)-1):
        for j in range(i+1, len(point_list)):
            
            dist_sq = (point_list[i][0] - point_list[j][0])**2 + (point_list[i][1] - point_list[j][1])**2  
            
            if dist_sq > max_dist:
                second_dist = max_dist 
                third_point = first_point
                fourth_point = second_point
                
                max_dist = dist_sq
                first_point = point_list[i]
                second_point = point_list[j]
            
            elif dist_sq <= max_dist and dist_sq > second_dist:
                second_dist = dist_sq
                third_point = point_list[i]
                fourth_point = point_list[j]
                
    return first_point, second_point, math.sqrt(max_dist), third_point, fourth_point, math.sqrt(second_dist)   

    
# if object is a circle, find max distance which also has the closest slope to the best fit slope 
def find_tiebreaker( x, y, best_slope, max_dist ):

    first_point = (x[0], y[0])
    second_point = (x[1], y[1])
    min_slope_diff = 999999.999
    
    for i in range(0, len(x)-1):
        for j in range(i+1,len(x)+i):
        
            if j >= len(x):
                j = j % len(x)
            
            dist = math.sqrt( (x[i] - x[j])**2 + (y[i] - y[j])**2 )
            if (max_dist - dist)/max_dist <= 0.08 and (x[i] - x[j]) != 0 :
                slope_diff = abs( (y[i] - y[j])/(x[i] - x[j]) - best_slope )
                
                if slope_diff < min_slope_diff:
                    min_slope_diff = slope_diff
                    first_point = (x[i], y[i])
                    second_point = (x[j], y[j])    

    return first_point, second_point
    

# find midpoints between two points given a set of fractions  
def get_midway (p1, p2, fractions):
    
    midpts = []
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    for n, fract in enumerate(fractions):
        x = p1[0] + fract * dx 
        y = p1[1] + fract * dy 
        midpts.append( (x,y) )
            
    return midpts

# find intersection of 2 lines     
def line_intersect(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    div = det(xdiff, ydiff)
    if div == 0:
        return (-999999,-999999) 

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y    
    
# find points of intersection between normal line and all possible line segments in contour 
def find_intersect( x1, y1, x2, y2):
    
    intersect = []
    
    for i in range(0,len(x2)):
        j = i + 1
        if j == len(x2):
            j = 0
        
        # find point of intersection
        point = line_intersect( ((x1[0], y1[0]), (x1[-1], y1[-1])), ((x2[i], y2[i]), (x2[j], y2[j])) )
        
        # check that the point lies on the contour 
        dotproduct = (point[0] - x2[i]) * (x2[j] - x2[i]) + (point[1] - y2[i])*(y2[j] - y2[i])
        squaredlength = (x2[j] - x2[i])**2 + (y2[j] - y2[i])**2
        
        if dotproduct >= 0 and dotproduct <= squaredlength and point not in intersect:
            intersect.append( point )

    return intersect

def canny(img):
    cimg = cv2.Canny(img,100,200)
    if np.amax(cimg,axis=(0,1)) == 0:
        cimg = cv2.Canny(img,20,100)
    if np.amax(cimg,axis=(0,1)) == 0:
        cimg = cv2.Canny(img,5,20)
    if np.amax(cimg,axis=(0,1)) == 0:
        cimg = cv2.Canny(img,1,5)   
    return cimg

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

def edge_detect( img ):
    edges1 = canny(img)

    edges2, rows, cols, min_x, min_y, max_x, max_y = fill_in_diagonals(edges1)
                    
    edges2 = edges2[min_y:max_y+1, min_x:max_x+1]
    row_dist = max_y+1-min_y
    col_dist = max_x+1-min_x
    scale = int(1000 / max(row_dist,col_dist))
    try:
        edges2 = cv2.resize(edges2, dsize=(row_dist*scale, col_dist*scale), interpolation=cv2.INTER_CUBIC)                
    except:
        edges2 = edges1
        
    return edges1, edges2 

def find_contours(img):
    img_gamma = adjust_gamma(img, .33)
    rows,cols = img.shape[:2]

    # create grayscale image and use Canny edge detection
    edges1, edges2 = edge_detect(img)
    edges_gamma, edges2_gamma = edge_detect(img_gamma) 
    
    # find contours from edge image
    try:
        image, contours, hierarchy = cv2.findContours(edges1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    except:
        contours, hierarchy = cv2.findContours(edges1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
    try:
        contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    except:
        image, contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            
    total_perimeter = 0
    for n, contour in enumerate(contours):
        total_perimeter += cv2.arcLength(contour,False)
        
    total_perimeter_gamma = 0
    for n, contour_g in enumerate(contours_gamma):
        total_perimeter_gamma += cv2.arcLength(contour_g,False)  
        
    if total_perimeter_gamma >= 1.1 * total_perimeter:
        contours = contours_gamma
        edges2 = edges2_gamma
        
    return contours, edges2

# functions to save and load databases
# the advantage of structuring all the methods in a similar way is that we can write looping code like this
def generate_databases(imgs, method_classes, name):
    for method_c in method_classes: # for each method class
        method = method_c() # create the method by instancing the class
        db = method.generate_database(imgs) # generate the database from the images
        filename = "db_"+method.__class__.__name__+"_"+name # this is what the database file should be called
        save_obj(db, filename) # save it
        
def load_databases(method_classes, name):
    loaded_methods = [] # list of loaded methods to return
    for method_c in method_classes: # for each method class
        filename = "db_"+method_c.__name__+"_"+name # this is what the database file should be called
        db = load_obj(filename) # load the database file
        method = method_c(db) # construct the instance of the method class using the database
        loaded_methods.append(method) # add it to the list of methods to return
    return loaded_methods


def plot_results (img, results, images, result_index = 1):
    # img is the original image
    # results is a list of (index, score1, score2...) tuples in descending order of match closeness, eg [(1, 100), (0, 77.7), (2, 15.6)]    
    # images is the set of images we're checking against
    
    # create plot of original image and best matches 
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(32, 32),sharex=False, sharey=False)
    ( (ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12) ) = ax
    result_cells = [ax3, ax4, ax5, ax6, ax9, ax10, ax11, ax12]

    # this part is just so that we can make the original image bigger than the others
    gs = ax[1,2].get_gridspec()
    ax1.remove()
    ax2.remove()
    ax7.remove()
    ax8.remove()
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.imshow(gray(img), cmap=plt.cm.gray)
    ax1.set_title('Query Image', fontsize=20, y = 1.0)

    # this shows the result images
    for c_inx in range(len(result_cells)):
        if c_inx >= len(images):
            break
        result_cells[c_inx].imshow(gray(images[results[c_inx][0]]), cmap=plt.cm.gray)
        result_cells[c_inx].set_xlim([0,32])
        result_cells[c_inx].set_ylim([32,0])
        result_cells[c_inx].set_title('match score: ' + '%.1f'%(results[c_inx][result_index]), fontsize=20, y = 1.0)
    
    # maximize the window and display plots 
    fig.tight_layout()
    plt.show()

def test_combined(methods, weights = []):
    # this version of test_combined should be able to take any number of methods
    # if the weights are not specified, will weigh everything equally
    match_combined = [] # this will have an entry [index, m1_score, m2_score,..., combined_score] for each image
    num_imgs = len(methods[0])
    num_methods = len(methods)
    for idx, entry in enumerate(zip(*methods)): # *methods 'unpacks' the list, basically stripping the outside pair of brackets
        # for each image we build up an entry [index, m1_score, m2_score,..., combined_score]
        
        # score serves as an accumulator for the score, we'll keep adding the score of this image
        # for each method to this, and then store the final result at the end of the entry
        score = 0 
        tmp = [idx] # this will be the entry, starts off with the image index
        for m_idx, method_score in enumerate(entry): # then for each method,
            tmp.append(method_score[1]) # we add the individual score to the entry
            score += method_score[1] * (weights[m_idx] if m_idx < len(weights) else 1) # and accumulate the weighted score
        tmp.append(score) # after we look at each method we add the combined weighted score to the entry
        match_combined.append(tmp) # and then we add the entry to the matched list
    match_combined = sorted(match_combined, key = lambda tup: tup[-1], reverse = True ) # sort by combined score 
    return match_combined

def run(methods, images, aberrations, candidates=None, weights=[]):
    # this version of run builds a dictionary containing stats for each (aberrated) image, rather than just collecting stats
    # in aggregate. This will then be piped into a dataframe so we can get any statistics we want.
    candidates = candidates or range(len(images))
    results = {} # this dictionary will be used to collect results to log
    scores = {} # this dictionary will collect information to run the logistic regression on
    for img_idx in candidates: # run through all the candidates
        img = images[img_idx]
        if img_idx % 10 == 0:
            print(img_idx) # this is here so we can tell how far along the run we are
        if img is None: # if on the odd chance some image is missing, skip it
            continue
        for aber in aberrations: # run through each of the aberrations for the image
            query_image = aber(img)
            method_lists = [] # we're going to gather the query lists of each of the methods here so we can combine them later
            for method_idx, method in enumerate(methods): # try each of the methods
                start_time = perf_counter() # this is a timestamp of when we start the method
                xl = method.run_query(query_image, candidates=candidates) # get the scores for all the images in the method's database
                method_lists.append(xl) # store the unsorted version for easy combining
                xl = sorted(xl, key = lambda y: y[1], reverse=True) # sort the results by score
                time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
                score = 0 # once we find the input image, we're gonna put the score of it in here
                rank = 0 # we're gonna count up to the rank of the input image
                for hit in xl: # go through all the scores
                    rank += 1
                    if hit[0] == img_idx: # stop when we've found the input image
                        score = hit[1]
                        break
                # now we store the data that will become a row of the dataframe
                # note that results.setdefault("blah", []) will either return results["blah"] if it's not None
                # or will return [], so we can just .append and not worry about initializing.
                results.setdefault("img_idx",[]).append(img_idx) # note that the image index doesn't matter for statistics
                results.setdefault("aberration",[]).append(aber.__name__)
                results.setdefault("method",[]).append(method.__class__.__name__)
                results.setdefault("score",[]).append(score)
                results.setdefault("rank",[]).append(rank)
                results.setdefault("time",[]).append(time_elapsed)
                # now we store the information for logistic regression
                scores.setdefault(method.__class__.__name__,[]).append(method_lists[-1])

            # Store which image was the correct one, for use in logistic regression
            scores.setdefault("labels",[]).append([1 if i == img_idx else 0 for i in candidates])
            # And also what was the true index, for seperating the training and test sets
            scores.setdefault("idx",[]).append([img_idx for i in candidates])

            # now it's time for the combined method
            start_time = perf_counter() # this is a timestamp of when we start the method
            combined_list = test_combined(method_lists, weights) # this comes out pre-sorted
            time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
            score = 0 # once we find the input image, we're gonna put the score of it in here
            rank = 0 # we're gonna count up to the rank of the input image
            for hit in combined_list: # go through all the scores
                rank += 1
                if hit[0] == img_idx: # stop when we've found the input image
                    score = hit[1]
                    break
            results.setdefault("img_idx",[]).append(img_idx) # note that the image index doesn't matter for statistics
            results.setdefault("aberration",[]).append(aber.__name__)
            results.setdefault("method",[]).append("combined_method")
            results.setdefault("score",[]).append(score)
            results.setdefault("rank",[]).append(rank)
            results.setdefault("time",[]).append(time_elapsed)

            # and again with equal weighting
            start_time = perf_counter() # this is a timestamp of when we start the method
            combined_list = test_combined(method_lists) # this comes out pre-sorted
            time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
            score = 0 # once we find the input image, we're gonna put the score of it in here
            rank = 0 # we're gonna count up to the rank of the input image
            for hit in combined_list: # go through all the scores
                rank += 1
                if hit[0] == img_idx: # stop when we've found the input image
                    score = hit[1]
                    break
            results.setdefault("img_idx",[]).append(img_idx) # note that the image index doesn't matter for statistics
            results.setdefault("aberration",[]).append(aber.__name__)
            results.setdefault("method",[]).append("uwcombined_method")
            results.setdefault("score",[]).append(score)
            results.setdefault("rank",[]).append(rank)
            results.setdefault("time",[]).append(time_elapsed)

    # save everything to file and return the dataframes
    results_pd = pd.DataFrame(data=results)
    scores_pd = pd.DataFrame(data=scores)
    log_num = len(glob("Logs/*")) + 1
    results_pd.to_csv ('Logs/results_'+str(log_num)+'.csv', index = None, header=True)
    score_num = len(glob("Training/*")) + 1
    scores_pd.to_csv ('Training/results_'+str(score_num)+'.csv', index = None, header=True)
    return results_pd, scores_pd

def chunks(x, n=10):
    for i in range(0, len(x), n):
        yield x[i:i+n]

def run_in_chunks(methods, images, aberrations, weights=[]):
    candidates = [i for i in range(len(images))]
    for candidate_chunk in chunks(candidates, 100):
        pass