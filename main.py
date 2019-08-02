
print "loading packages..."

import numpy as np
import tensorflow as tf
import csv, random, sys, time#, copy 

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
		linesMod = [x+1 for x in lines]
		for lineNum, line in enumerate(data_file):
			if lineNum in linesMod:
				image = []
				newLine = line.strip().split(',')
				newLine = map(int, newLine)
				labels.append(newLine[0])
				#fix this, inefficient 
				newLine = [[x] for x in newLine]
				#channelLine = []
				#for val in newLine:
				#	channelLine.append([newLine[val]])
				#newLine = channelLine
				#end of stuff to fix
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

			
def runNet( isTrain, file, image_dim, image_count, batch_size, train_epochs, num_classes, learning_rate, model ):
	
	#input data and labels
	image = tf.placeholder(shape=(1,image_dim[0],image_dim[1],image_dim[2]), dtype=tf.float32, name='image')
	label = tf.placeholder(shape=(1,10), dtype=tf.float32, name='label')

	#default order NHWC, HWInOut for filters

	#weights
	wc1 = tf.Variable(tf.random_normal([5, 5, 1, 16]), name='wc1')
	wc2 = tf.Variable(tf.random_normal([5, 5, 16, 32]), name='wc2')
	wl1 = tf.Variable(tf.random_normal([1568, 256]), name='wl1') #[32 * 4 * 4, 256]
	wl2 = tf.Variable(tf.random_normal([256, 10]), name='wl2')

	#biases
	bc1 = tf.Variable(tf.random_normal([16]), name='bc1')
	bc2 = tf.Variable(tf.random_normal([32]), name='bc2')
	bl1 = tf.Variable(tf.random_normal([256]), name='bl1')
	bl2 = tf.Variable(tf.random_normal([10]), name='bl2')	
	
	#define net flow
	#conv layer 1
	x = tf.nn.conv2d(image, wc1, strides=[1,1,1,1], padding="SAME")
	x = tf.nn.relu(tf.nn.bias_add(x, bc1))
	x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#conv layer 2
	x = tf.nn.conv2d(x, wc2, strides=[1,1,1,1], padding="SAME")
	x = tf.nn.relu(tf.nn.bias_add(x, bc2))
	x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#fully connected layer
	x = tf.reshape(x, [-1,1568])
	x = tf.sigmoid(tf.add(tf.matmul(x, wl1), bl1))
	x = tf.nn.dropout(x, 0.75)
	classif = tf.sigmoid(tf.add(tf.matmul(x, wl2), bl2))
	
	#loss and error functions
	squaredError = tf.square(label - classif)
	loss = tf.reduce_sum(squaredError)
	
	#training operation
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train = optimizer.minimize(loss)
	
	#traning session
	if isTrain == True:
		model = []
		with tf.Session() as sess:
			sess.run( tf.global_variables_initializer() )
			print "loading data..."
			img, labl = readImages( file, range(image_count), image_dim, image_count)
			print "training model..."
			maxBatchID = image_count // batch_size
			batchID = 0
			startTime = time.time()
			trainTime = 0
			dataTime = 0
			for ep in range(train_epochs):
				image_adr = range(image_count)
				random.shuffle(image_adr)
				print "training: epoch " + str(ep + 1) + "..." 
				countImage = -1
				#while batchID < maxBatchID:
					#img, labl = readImages( file, image_adr[(batch_size*batchID):(batch_size*(batchID+1))], image_dim, image_count)
					#batchID += 1
				print ""
				for image_num in image_adr:
					countImage += 1
					if countImage % 100 == 0:
						sys.stdout.write("\033[F")
						print str((100 * countImage) // image_count) + "%" + "   total time: " + str(time.time() - startTime) + "   train time: " + str(trainTime) 
						#sys.stdout.write("\033[F")
					#[img], [labl] = readImages( file, [image_num], image_dim, image_count)
					#print "Image Num:", image_num
					#print labl
					#print labl[image_num]
					lab = convertLabel(labl[image_num], num_classes)
					#print lab
					trainStartTime = time.time()
					sess.run(train, feed_dict={image: [img[image_num]], label: [lab]})
					trainTime += (time.time() - trainStartTime)
				sys.stdout.write("\033[F")
				print "                                                                  " 
				sys.stdout.write("\033[F")
				#for batchID in range(maxBatchID):
				#	batchRows = rows[(batchID * batchSize) : ((batchID + 1) * batchSize)]
				#	batchLabels = labels[(batchID * batchSize) : ((batchID + 1) * batchSize)]
				#	print str((100*batchID) // maxBatchID) + "%"
				#	sys.stdout.write("\033[F")
				#	for adr in range(batchSize):
						#sess.run(train, feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
					#print "Squared_Error", sess.run(squaredError,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
					#print "Loss", sess.run(loss,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
					#print "Class", sess.run(classif,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
					#print "Label", sess.run(label,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
					#print "batch:", batchID + 1
			#print "calculating MSE..."
			model = [sess.run(wc1), sess.run(wc2), sess.run(wl1), sess.run(wl2)], [sess.run(bc1), sess.run(bc2), sess.run(bl1), sess.run(bl2)]
			#MSE, _ = testModel( labels, rows, model, hidden)	
			#sys.stdout.write("\033[F")
			#print "MSE:", MSE[0][0]
		return model
		
	#testing session
	else: 
		loadWeight, loadBias = model
		totalError = 0
		results = []
		with tf.Session() as sess:
			sess.run( tf.global_variables_initializer() )
			print "loading data..."
			img, labl = readImages( file, range(image_count), image_dim, image_count)
			print "testing model..."
			image_adr = range(image_count)
			for image_num in image_adr:
				lab = convertLabel(labl[image_num], num_classes)
				totalError += sess.run(squaredError, feed_dict={image: [img[image_num]], label: [lab], \
				wc1: loadWeight[0], wc2: loadWeight[1],	wl1: loadWeight[2],	wl2: loadWeight[3], bc1: loadBias[0], \
				bc2: loadBias[1], bl1: loadBias[2], bl2: loadBias[3]})
				results.append([ sess.run(label, feed_dict={image: [img[image_num]], label: [lab], \
				wc1: loadWeight[0], wc2: loadWeight[1],	wl1: loadWeight[2],	wl2: loadWeight[3], bc1: loadBias[0], \
				bc2: loadBias[1], bl1: loadBias[2], bl2: loadBias[3]}), \
				sess.run(classif, feed_dict={image: [img[image_num]], label: [lab], \
				wc1: loadWeight[0], wc2: loadWeight[1],	wl1: loadWeight[2],	wl2: loadWeight[3], bc1: loadBias[0], \
				bc2: loadBias[1], bl1: loadBias[2], bl2: loadBias[3]})])
		return totalError / image_count, results


fashionmnist = "./fashionmnist/fashion-mnist_train.csv"
image_dim = [28, 28, 1]
batch_size = 1000
train_epochs = 1
learning_rate = 0.01
training_momentum = 0.5
num_classes = 10
# [outputChannels, kernelSize]
convolutions = [[10,3],[10,3],[10,3]]
fully_connected = [256, num_classes]


#print "counting images..."
image_count = countLines(fashionmnist)

#image_count = 1000

#train model
model = runNet(True, fashionmnist, image_dim, image_count, batch_size, train_epochs, num_classes, learning_rate, None)

#test model
results = runNet(False, fashionmnist, image_dim, image_count, batch_size, train_epochs, num_classes, learning_rate, model)

print "Average squared error:", results

print "writing model to file..."
out_file = open('model.txt', 'w')
out_file.write(str(model))
out_file.close


#print "testing convolution..."
#test_image = readImages(fashionmnist, [1], image_dim, image_count)[0][0]


#print "testing readImages..."
#print "number of images:", readImages(fashionmnist, [1], image_dim, image_count)[0][0]
#print "done"

	
