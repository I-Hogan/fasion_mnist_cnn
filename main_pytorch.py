

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv, random#, sys, copy 

from PIL import Image



#counts images in dataset
def countLines( file ):
	imageCount = -1
	with open(file, "r") as data_file:
		count_reader = csv.reader(data_file, delimiter=',')
		for lineNum, line in enumerate(data_file):
			imageCount += 1 
	return imageCount

#reads and returns images from fasionmnist in lines
def readImages( file, lines, image_dim, image_count ):
	images = []
	labels = []
	linesMod = [x % image_count for x in lines]
	with open(file, "r") as data_file:
		for lineNum, line in enumerate(data_file):
			if lineNum in linesMod:
				image = []
				newLine = line.strip().split(',')
				newLine = map(int, newLine)
				labels.append(newLine[0])
				for y in range(image_dim[1]):
					offset = image_dim[0] * y + 1
					image.append(newLine[offset:(offset+image_dim[0])])
				images.append(image)
	#show image
	#image = np.asarray(images[0])
	#img = Image.fromarray(image.astype('uint8'))
	#img.show()
	return images, labels
	
def convertLabel( label, num_classes ):
	label_array = [0] * num_classes
	label_array[label] = 1
	return label_array
	

class fashionmnistCNN(nn.Module):
	def __init__(self, image_dim, convolutions, fully_connected):
		super(fashionmnistCNN,self).__init__()
		self.conv = []
		self.fc = []
		outputChannels = 0
		for conv in range(len(convolutions)):
			if conv == 0:
				inputChannels = image_dim[2]
			else:
				inputChannels = convolutions[conv-1][0]
			outputChannels = convolutions[conv][0]
			kernelSize = convolutions[conv][1]
			self.conv.append(nn.Conv2d(inputChannels, outputChannels, kernel_size=kernelSize, stride=1, padding = 0))
		self.reshape_dim = ((image_dim[0] / (len(self.conv) * 2)) * (image_dim[1] / (len(self.conv) * 2)) * outputChannels)
		for fc in range(len(fully_connected)): 
			if  fc == 0:
				inputFeatures = self.reshape_dim #((image_dim.x * image_dim.y) / (len(self.conv) * 2)) * outputChannels
			else:
				inputFeatures = fully_connected[fc - 1]
			outputFeatures = fully_connected[fc]
			self.fc.append(nn.Linear(inputFeatures,outputFeatures))
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		
	def forward(self, input):
		for conv in range(len(self.conv)):
			x = F.relu(self.conv[conv](x))
			x = self.pool(x)
		x = x.view(-1, self.reshape_dim)
		for fc in range(len(self.fc)):
			if fc == len(self.fc - 1):
				x = self.fc[fc]
			else:
				x = F.relu(self.fc[fc])
		return x
			
def train( file, image_dim, image_count, batch_size, train_epochs, num_classes ):
	#model.train()
	for ep in range(train_epochs):
		image_adr = range(image_count)
		random.shuffle(image_adr)
		#for batch_num in range(image_count // batch_size):
		#	batch, batchLables = readImages( file, range((batch_num * batch_size), ((batch_num + 1) * batch_size)), image_dim, image_count )
		for image_num in image_adr:
			[image], [label] = readImages( file, [image_num], image_dim, image_count)
			label = convertLabel(label, num_classes)
			
			
def convTest(test_image):
	conv1 = nn.Conv2d(1,1,3,stride=1)
	
	image = torch.tensor(test_image)
	kernel = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]])
	
	x = torch.randn(1,1,5,5);
	y = torch.randn(1,1,3,3);
	z = F.conv2d(x,y);
	
	print z
	
	image = F.conv2d(image,kernel)
	
	
	print image
	


fashionmnist = "./fashionmnist/fashion-mnist_train.csv"
image_dim = [28, 28, 1]
batch_size = 256
train_epochs = 1
learning_rate = 0.01
training_momentum = 0.5
num_classes = 10
# [outputChannels, kernelSize]
convolutions = [[10,3],[10,3],[10,3]]
fully_connected = [256, num_classes]


print "counting images..."
image_count = countLines(fashionmnist)

model = fashionmnistCNN(image_dim, convolutions, fully_connected)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=training_momentum)
#train( fashionmnist, image_dim, image_count, batch_size, train_epochs, num_classes)

print "testing convolution..."
test_image = readImages(fashionmnist, [1], image_dim, image_count)[0][0]
convTest(test_image)


#print "testing readImages..."
#print "number of images:", readImages(fashionmnist, [1], image_dim, image_count)[0][0]
#print "done"

	
