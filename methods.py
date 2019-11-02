# these are the imports needed to run the methods
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import math
import mahotas
from sklearn.preprocessing import normalize

from icon_util import *

class method_base(object):
    # base class that all methods inherit from
    # this is just to avoid having to rewrite all the stuff that is the same between
    # the various methods

    # The database stores the image descriptors for all the known images,
    # indexed by the image index.
    # Either initialize as an empty dict or use the one passed in
    def __init__(self, database = None):
        self.database = database or {}

    # Take in an image and return its descriptor calculated by the method.
    # Each method will override this.
    def create_query(self, img, **kwargs):
        return 1

    # Take in two descriptors and say how similar they are.
    # Each method will override this.
    def compare_queries(self, v1, v2, **kwargs):
        return 1

    # If the method needs to normalize its results somehow, do that by
    # overriding this function.
    def normalize_results(self, results, **kwargs):
        return results

    # The last three functions should be the same for all methods.

    # Take an image descriptor, and optionally a set of indices for images it's likely to be
    # and return a list of (index, similarity) pairs for each candidate image,
    # or for every known image if candidates are not given.
    def test_query(self, img_vec, candidates=None, **kwargs):
        candidates = candidates or self.database.keys()
        results = [(img_index, self.compare_queries(img_vec, self.database[img_index], **kwargs)) for img_index in candidates]
        return self.normalize_results(results)

    # Just a convenience function to get an image descriptor and run the comparison.
    def run_query(self, img, **kwargs):
        img_query = self.create_query(img, **kwargs) # create the image descriptor
        results = self.test_query(img_query, **kwargs) # compare against all the images in the database
        return results

    # Make and store image descriptors for every given image.
    def generate_database(self, images, **kwargs):
        self.database  = {}
        for img_index in range(len(images)):
            if img_index%50 == 0:
                print(img_index)
            img = images[img_index]
            self.database[img_index] = self.create_query(img, **kwargs)
        return self.database

# neural
class neural_method(method_base):
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    #strip the final layer to get a feature vector
    model = nn.Sequential(*list(model.children())[:-1])  
    # Set model to evaluation mode
    model.eval()

    def create_query(self, img, **kwargs):
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        pillize = transforms.ToPILImage()
        loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
        l_img = loader(img)
        t_img = Variable(l_img).unsqueeze(0)
        f_vec = self.model(t_img).squeeze()
        n_vec = f_vec.detach().numpy()
        n_vec = n_vec / np.linalg.norm(n_vec)
        return n_vec

    def compare_queries(self, v1, v2, **kwargs):
        cos = np.dot(v1, v2)
        sim = (cos + 1) * 50
        return sim.item()
    
class euclidean_neural_method(method_base):
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    #strip the final layer to get a feature vector
    model = nn.Sequential(*list(model.children())[:-1])  
    # Set model to evaluation mode
    model.eval()
    
    def create_query(self, img, **kwargs):
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        pillize = transforms.ToPILImage()
        loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
        l_img = loader(img)
        t_img = Variable(l_img).unsqueeze(0)
        f_vec = self.model(t_img).squeeze()
        n_vec = f_vec.detach().numpy()
        return n_vec
    
    def compare_queries(self, v1, v2, **kwargs):
        sim = 100/(np.linalg.norm(v1-v2)+1)
        return sim
    
class trained_neural_method(euclidean_neural_method):
    model = models.resnet18()
    model.fc = nn.Linear(512,8)
    model.load_state_dict(torch.load("models/EmbeddingIconResnet.pt"))
    model.eval()

class small_neural_method(method_base):
    # this is the definition of the custom neural network
    class IconEmbeddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.convnet = nn.Sequential(
                nn.Conv2d(3, 32, 5),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, 5),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2))
            self.fc = nn.Sequential(
                nn.Linear(64 * 4 * 4, 256),
                nn.PReLU(),
                nn.Linear(256, 256),
                nn.PReLU(),
                nn.Linear(256, 8)
                )
        def forward(self, x):
            output = self.convnet(x)
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
            return output

        def get_embedding(self, x):
            return self.forward(x)
        
    model = IconEmbeddingNet()
    model.load_state_dict(torch.load("models/IconEmbeddingNet.pt"))
    model.eval()
    
    def create_query(self, img, **kwargs):
        scaler = transforms.Resize((28, 28))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        pillize = transforms.ToPILImage()
        loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
        l_img = loader(img)
        t_img = Variable(l_img).unsqueeze(0)
        f_vec = self.model(t_img).squeeze()
        n_vec = f_vec.detach().numpy()
        return n_vec

    def compare_queries(self, v1, v2, **kwargs):
        sim = 100.0/(np.linalg.norm(v1-v2)+1)
        return sim
    

# orb
class orb_method(method_base):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def create_query(self, img, **kwargs):
        img = gray(img)
        img = cv2.resize(img,None, fx=13, fy=13, interpolation=cv2.INTER_AREA)

        # find the keypoints and descriptors with orb
        kp1, des1 = self.orb.detectAndCompute(img,None)
        if len(kp1) < 2:
            des1 = None
        return des1

    def compare_queries(self, img_kp1, img_kp2, **kwargs):
        if img_kp1 is None or img_kp2 is None:
            return 0
        try:
            matches = self.matcher.match(img_kp1,img_kp2)
            score = len(matches)
            return score
        except:
            return 0

    def normalize_results(self, results, **kwargs):
        # find the result with the highest score, then divide everything
        # accordingly so that score is 100
        best = 1
        for i in results:
            if i[1] > best:
                best = i[1]
        results = [(x[0],round(100 * x[1]/best)) for x in results]
        return results


class sift_method(method_base):
    # and define all the auxilliary stuff needed for the method to run
    # parameters, objects, etc...
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params,search_params)

    def create_query(self, img, **kwargs):
        # preproccess the image
        img = gray(img)
        img = cv2.resize(img,None, fx=13, fy=13, interpolation=cv2.INTER_AREA)

        # find the keypoints and descriptors with sift
        kp1, des1 = self.sift.detectAndCompute(img,None)
        # if sift doesn't give us enough info to match, return none
        if len(kp1) < 2:
            des1 = None
        return des1

    def compare_queries(self, img_kp1, img_kp2, **kwargs):
        # compare the queries generated by two images to obtain a similarity score

        # if either of the two image queries is missing, then return 0
        if img_kp1 is None or img_kp2 is None:
            return 0
        # try to run the matcher, if it fails return 0
        try:
            matches = self.matcher.knnMatch(img_kp1, img_kp2, k=2)
            # count up the number of matches that are "close enough"
            score = 0
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    score += 1
            return score
        except:
            return 0

    def normalize_results(self, results, **kwargs):
        # find the result with the highest score, then divide everything
        # accordingly so that score is 100
        best = 1
        for i in results:
            if i[1] > best:
                best = i[1]
        results = [(x[0],round(100 * x[1]/best)) for x in results]
        return results


class zernike_method(method_base):
    def create_query(self, img, **kwargs):
        _, _, min_x, min_y, max_x, max_y, edges1 = image_preprocess(img) 
        ## create zernike vector 
        edges2 = edges1[min_y:max_y+1, min_x:max_x+1]
        edges2 = cv2.resize(edges2, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        zernike = mahotas.features.zernike_moments(edges2, 16)  
        return normalize(zernike[:,np.newaxis], axis=0).ravel()

    def compare_queries(self, x,y, **kwargs):
        dot_prod = sum(i[0] * i[1] for i in zip(x, y))
        return 50.0 * (dot_prod + 1.0) 


class contour_method(method_base):
    fractions = [.1,.2,.3,.4,.5,.6,.7,.8,.9]    
    
    def create_query(self, img, fractions=None, **kwargs):
        rows,cols = img.shape[:2]
        contours, edges = find_contours(img)
        dist_append = []
        
        fractions = fractions or self.fractions

        # find hull and defects of the contour 
        for n in range(0,len(contours)):
            cnt = contours[n]
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

                    if n == 0:
                        for n1 in range(0,len(contours)):
                            cnt1 = contours[n1]    

                            # find points of intersection between normal line and contour, plot those points
                            intersect = find_intersect(x, y, cnt1[:,0,0], cnt1[:,0,1])

                            # distance between middle point and intersect points, normalize using length of longest line
                            dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / max_dist , 3 ) for i in intersect ]
                            for d in dist:
                                if (fractions[m],d) not in dist_append:
                                    dist_append.append((fractions[m],d))

                    else:
                        # find points of intersection between normal line and contour, plot those points
                        intersect = find_intersect(x, y, cnt[:,0,0], cnt[:,0,1])

                        # distance between middle point and intersect points, normalize using length of longest line
                        dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / max_dist , 3 ) for i in intersect ]
                        for d in dist:
                            if (fractions[m],d) not in dist_append:
                                dist_append.append((fractions[m],d))

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
                            for d in dist:
                                if (fractions[m],d) not in dist_append:
                                    dist_append.append((fractions[m],d))              

        ## create final dictionary for contour method distances    
        query = {}        
        for d in dist_append:
            query.setdefault(d[0],[]).append(d[1])    

        return query         

    def compare_queries(self, query1, query2, error=0.002, fractions=None, **kwargs):
        fractions = fractions or self.fractions
        total_points = 0
        matched = 0
        reverse_matched = 0
        
        # loop through each of the fractions
        for f in fractions:
            for dist in query2.get(f):
                total_points += 1
                for query_dist in query1.get(f):
                    if query_dist >= dist - error and query_dist <= dist + error:
                        matched += 1
                        break

                for query_dist in query1.get(round(1-f,1)):
                    if query_dist >= dist - error and query_dist <= dist + error:
                        reverse_matched += 1
                        break

        matched = max(matched, reverse_matched)
        if total_points > 0:
            score = round( float( 100 * matched/total_points ), 1 )
        else:
            score = 0
            
        return score