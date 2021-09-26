import cv2
import numpy as np
import math
## all the functions related to zernike and contour
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
    img = gray(img)
    rows,cols = img.shape[:2]
    # create grayscale image and use Canny edge detection
    cimg = canny(img)   
    
    dcimg, rows, cols, min_x, min_y, max_x, max_y = fill_in_diagonals(cimg)
    
    return rows, cols, min_x, min_y, max_x, max_y, dcimg    

def prep_img(img,mgray=True):
    if mgray:
        img = gray(img)
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

def split_contour(x_cnt, y_cnt, x_mid, y_mid):
    
    if x_cnt > x_mid:
        return 0
    elif x_cnt == x_mid and y_cnt > y_mid:
        return 0
    else:
        return 0.5

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

def edge_detect(img):
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
    gamma_list = [0.2, 0.33, 0.5, 1, 1.5, 3, 5]
    perimeter_list = []
    
    for g in range(0,len(gamma_list)):

        img_gamma = adjust_gamma(img, gamma_list[g])
    
        # create grayscale image and use Canny edge detection
        edges_gamma, edges2_gamma = edge_detect(img_gamma) 
        
        # use ellipse dilation to fill the gaps in countours  
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edges_gamma = cv2.dilate(edges_gamma, kernel)
        edges2_gamma = cv2.dilate(edges2_gamma, kernel)
        
        # find contours from edge image
        # the try,except is because some versions of opencv return the image as the first parameter
        try:
            contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        except:
            image, contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            
        # use 10 biggest countours      
        contours_gamma = sorted(contours_gamma, key = lambda x:cv2.contourArea(x), reverse = True)[:5]  
        total_perimeter_gamma = 0
        for n, contour_g in enumerate(contours_gamma):
            total_perimeter_gamma += cv2.arcLength(contour_g,False)   

        perimeter_list.append(total_perimeter_gamma)  
        
    g_max = np.argmax(perimeter_list)   
    img_gamma = adjust_gamma(img, gamma_list[g_max]) 
    edges_gamma, edges2_gamma = edge_detect(img_gamma)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges_gamma = cv2.dilate(edges_gamma, kernel)
    try:
        contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    except:
        image, contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            
    contours_gamma = sorted(contours_gamma, key = lambda x:cv2.contourArea(x), reverse = True)[:5]  
      
    return contours_gamma, edges2_gamma

def frange(start, stop, step):
    i = start
    while i < stop:
        if i == int(i):
            i = int(i)
        yield i
        i += step

        
def do_contour_create(img,fractions,**kwargs):
    rows, cols = img.shape[:2]
    if rows < 130:
        img = prep_img(img)

    rows, cols = img.shape[:2]    

    contours, edges = find_contours(img)
    dist_append = []

    # for each contour
    for n in range(0,len(contours)):
        cnt = contours[n]
        # find hull and defects of the contour 
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)

        # initialize list to track all points on the hull as well as points with defects 
        point_track = []

        for j in range (hull.shape[0]):    
            i = hull[j]
            point = tuple(cnt[i][0][0])
            point_track.append(point)

        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                far = tuple(cnt[f][0])
                point_track.append(far)

        # 4b. find max distance and plot 2 points along with longest line 
        if len(point_track) >= 2:
            [point1, point2, max_dist, point3, point4, second_dist] = find_max(point_track)
            ratio_dist = second_dist/max_dist

            # find midway points along line segment  
            midpts = get_midway( point1, point2, fractions ) 

            # slope of normal line 
            if (point2[0]-point1[0]) == 0:
                slope = 0
            elif (point2[1]-point1[1]) == 0:
                slope = 10000
            else:
                slope = -1 / ( (point2[1]-point1[1]) / (point2[0]-point1[0]) )

            # 4d. loop through the middle/quarter/eighth points and plot the normal lines 
            for m, pt in enumerate(midpts):

                x = np.linspace(0.0,cols,3)
                y = [float(slope * i - slope * pt[0] + pt[1]) for i in x]  

                intersect = find_intersect(x, y, cnt[:,0,0], cnt[:,0,1])

                # distance between middle point and intersect points, normalize using length of longest line
                dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / max_dist , 3 ) for i in intersect ]
                split = [split_contour(i[0],i[1],pt[0],pt[1]) for i in intersect]

                for d in range(0,len(dist)):
                    if (n+split[d],fractions[m],dist[d]) not in dist_append:
                        dist_append.append((n+split[d],fractions[m],dist[d]))

            # find midway points along second best line segment  
            if n <= 1 and ratio_dist >= .988:
                midpts2 = get_midway( point3, point4, fractions ) 

                # slope of normal line 
                if (point4[0]-point3[0]) == 0:
                    slope2 = -1 / ( (point4[1]-point3[1]) / (point4[0]-point3[0] + 0.0001) )
                elif (point4[1]-point3[1]) == 0:
                    slope2 = -1 / ( (point4[1]-point3[1] + 0.0001) / (point4[0]-point3[0]) )
                else:
                    slope2 = -1 / ( (point4[1]-point3[1]) / (point4[0]-point3[0]) ) 

                for m, pt in enumerate(midpts2):
                    x1 = np.linspace(0,cols,num=cols+1)
                    y1 = [slope2 * i - slope2 * pt[0] + pt[1] for i in x1] 

                    intersect = find_intersect(x1, y1, cnt[:,0,0], cnt[:,0,1])
                    if len(intersect) > 0:

                        # distance between middle point and intersect points, normalize using length of longest line
                        dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / second_dist , 2 ) for i in intersect ]
                        split = [split_contour(i[0],i[1],pt[0],pt[1]) for i in intersect]
                        for d in range(0,len(dist)):
                            if (5+split[d],fractions[m],dist[d]) not in dist_append:
                                dist_append.append((5+split[d],fractions[m],dist[d]))              

    ## create final dictionary for contour method distances    
    query = {}        
    for d in dist_append:
        query.setdefault(str((d[0],d[1])),[]).append(d[2])    

    return query
def do_contour_compare(query1,query2,fractions,error,**kwargs):
    exclude_circle = []

    # check if outer contour is circular
    for n in [0,0.5,5,5.5]:
        scores_circle = []
        for f in fractions:
            diff_circle = []
            for dist in query1.get(str((n,f)) , []):
                f1 = 2*abs(0.5 - f)                    
                diff_circle1 = abs(dist - np.sin(np.arccos(f1))/2)
                if dist > 0 :
                    diff_circle.append(diff_circle1)
            diff_circle.sort()
            if len(diff_circle) >= 1:
                partial_score_circle = max(100 - diff_circle[0]*100/error , 0)
            else:
                partial_score_circle = 0
            scores_circle.append(partial_score_circle)

        final_score_circle = sum(scores_circle)/len(scores_circle)
        if final_score_circle >= 90:
            exclude_circle.append(n)

    # compare each contour between query and database image, exclude outer circle 
    contour_matrix = []
    for n in frange(0,6,0.5):
        temp_matrix = []
        if n not in exclude_circle:             
            for m in frange(0,6,0.5):
                scores = []
                for f in fractions:          
                    diff = []
                    for dist in query1.get(str((n,f)) , []):
                        for query_dist in query2.get(str((m,f)) , []):
                            diff1 = abs(dist - query_dist)
                            if dist > 0 and query_dist > 0:
                                diff.append(diff1)
                    if len(diff) == 1:
                        partial_score = max(100 - diff[0]*100/error , 0)
                    elif len(diff) == 2:
                        diff.sort()
                        partial_score = max(100 - diff[0]*50/error - diff[1]*50/error, 0)
                    elif len(diff) > 2:
                        partial_score = max(100 - diff[0]*33.33/error - diff[1]*33.333/error - diff[2]*33.33/error, 0)
                    else:
                        partial_score = 0
                    scores.append(partial_score)

                temp_matrix.append( round(sum(scores)/len(scores),3))

        contour_matrix.append(temp_matrix)

    all_scores = []
    for i in range(len(contour_matrix)):
        try:
            temp = max(contour_matrix[i])
        except:
            temp = 0
        all_scores.append(temp)  

    if len(all_scores) > 0:
        final_score = sum(all_scores)/len(all_scores)
    else:
        final_score = 0



    # compare each contour between query and reversed database image, exclude outer circle
    contour_matrix = []
    for n in frange(0,6,0.5):  
        temp_matrix = []
        if n not in exclude_circle:
            for m in frange(0,6,0.5):
                scores = []
                for f in fractions:          
                    diff = []
                    for dist in query1.get(str((n,f)) , []):
                        for query_dist in query2.get(str((m,(1-f))) , []):
                            diff1 = abs(dist - query_dist)
                            if dist > 0 and query_dist > 0:
                                diff.append(diff1)
                    if len(diff) == 1:
                        partial_score = max(100 - diff[0]*100/error , 0)
                    elif len(diff) == 2:
                        diff.sort()
                        partial_score = max(100 - diff[0]*50/error - diff[1]*50/error, 0)
                    elif len(diff) > 2:
                        partial_score = max(100 - diff[0]*33.33/error - diff[1]*33.333/error - diff[2]*33.33/error, 0)
                    else:
                        partial_score = 0
                    scores.append(partial_score)

                temp_matrix.append( round(sum(scores)/len(scores),3))

        contour_matrix.append(temp_matrix)

    all_scores = []
    for i in range(len(contour_matrix)):
        try:
            temp = max(contour_matrix[i])
        except:
            temp = 0
        all_scores.append(temp)  

    if len(all_scores) > 0:
        reverse_score = sum(all_scores)/len(all_scores)
    else:
        reverse_score = 0

    return max(final_score, reverse_score)
