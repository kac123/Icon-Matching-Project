import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np

class neural(object):
	database = {}
	# Load the pretrained model
	model = models.resnet18(pretrained=True)
	#strip the final layer to get a feature vector
	model = nn.Sequential(*list(model.children())[:-1])  
	# Set model to evaluation mode
	model.eval()

	def __init__(self, database = None):
		if database:
			self.database = database

	def create_query(self, img):
		scaler = transforms.Resize((224, 224))
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
			std=[0.229, 0.224, 0.225])
		to_tensor = transforms.ToTensor()
		pillize = transforms.ToPILImage()
		loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
		l_img = loader(img)
		t_img = Variable(l_img).unsqueeze(0)
		f_vec = self.model(t_img).squeeze()
		return f_vec.detach().numpy()

	def compare_queries(self, v1, v2):
		sim = np.dot(v1, v2)
		return sim.item()

	def test_query(self, img_vec, candidates=None):
		neural_list = []
		if candidates is None:
			candidates = self.database.keys()
		for img_index in candidates:
			neural_list.append((img_index, self.compare_queries(img_vec, self.database[img_index])))
		return neural_list

	def generate_database(self, images):
		self.database  = {}
		for img_index in range(len(images)):
			if img_index%50 == 0:
				print(img_index)
			img = images[img_index]
			self.database[img_index] = self.create_query(img)
		return self.database