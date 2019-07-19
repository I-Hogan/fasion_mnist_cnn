


#import tensorflow as tf
import numpy as np
import csv, copy, random, math, sys

image_x = 28
image_y = 28

#reads and returns images from fasionmnist in lines
def readImages( file, lines ):
	images = []
	imageCount = -1
	with open(file, "r") as data_file:
		count_reader = csv.reader(data_file, delimiter=',')
		for line in count_reader:
			imageCount += 1 
	linesMod = [x % imageCount for x in lines]
	with open(file, "r") as data_file:
		data_reader = csv.reader(data_file, delimiter=',')
		labels = next(data_reader)
		lineNum = 0
		for line in data_reader:
			image = []
			lineNum += 1
			newLine = map(int, line)
			if lineNum in linesMod:
				for y in range(image_y):
					offset = image_x * y
					image.append(newLine[offset:(offset+image_x)])
				images.append(image)
		#if lineNum >= maxLines:
		#	data = data[:maxLines]
	return images


print "Tests:"
print readImages("../../Data/fashionmnist/fashion-mnist_train.csv", [1]) 
	
