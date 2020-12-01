import sys
from matplotlib import pyplot as plt
import numpy as np
import six
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Input,Dense,Flatten,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from autoencoder import Encoder,getInfo,createNParray,DataProcess1,DataProcess2
import ast


def getInfoLabel(f):
	temp = f.read(4) #pass first 4 bytes that are the magic number

	temp = f.read(4) #read 4 bytes
	temp = np.frombuffer(temp,dtype = np.uint8).astype(np.int64) #convert the string that stored in temp into an array(numpy) of integers
			#temp array has 4 integers, each integer represents a byte of number number_of_images
	img_number = (temp[0]<<24)|(temp[1]<<16)|(temp[2]<<8)|temp[3] #convert the temp array into an integer

	return img_number


#Creating NumPy Array with labels from file with file descriptor f
def createNParrayLabel(f,image_number):
	e = f.read(image_number*1)     #at the beginning e is string with labels
	e = np.frombuffer(e,dtype = np.uint8).astype(np.int64)  #convert string to numpy array of integers

	return e


def fullyConnected(encoded,filtersFC):
    fltn = Flatten()(encoded)
    fc = Dense(filtersFC, activation='relu')(fltn)      #fully connected layer
    outClassifier = Dense(10, activation='softmax')(fc) #output layer

    return outClassifier


def main():

	for pos in range(1,len(sys.argv)-1,2): #take values in range [1,len(sys.argv)-2] with step 2
		if sys.argv[pos] == '-d': #if this argument is '-d'
			training_set = sys.argv[pos+1]  #then the next argument will be training set
		if sys.argv[pos] == '-dl':
			training_labels = sys.argv[pos+1]
		if sys.argv[pos] == '-t':
			test_set = sys.argv[pos+1]
		if sys.argv[pos] == '-tl':
			test_labels = sys.argv[pos+1]
		if sys.argv[pos] == '-model':
			autoencoderModel = sys.argv[pos+1]

	#Read hyperparameters that we use in autoencoder:
	fd = open("info.txt","r")
	newlist = [line.rstrip() for line in fd.readlines()]
	numOfLayers = int(newlist[0])
	x_filter = int(newlist[1])
	y_filter = int(newlist[2])
	res = newlist[3]
	filtersPerLayer = ast.literal_eval(res)
	epochs = int(newlist[4])
	batch_size = int(newlist[5])
	ConvLayersEnc = int(newlist[6])
	ConvLayersDec = int(newlist[7])
	fd.close()

	#training_set preprocessing:
	fd1 = open(training_set,"rb")
	image_number,rows_number,columns_number = getInfo(fd1) #Getting info about Dataset
	e = createNParray(fd1,image_number,rows_number,columns_number) #Creating NumPy Array
	fd1.close()
	e = DataProcess1(e,rows_number,columns_number) #Data Processing
	trainingData = DataProcess2(e) #Data Processing

	#test_set preprocessing:
	fd2 = open(test_set,"rb")
	image_number,rows_number1,columns_number1 = getInfo(fd2) #Getting info about Dataset
	e = createNParray(fd2,image_number,rows_number1,columns_number1) #Creating NumPy Array
	fd2.close()
	e = DataProcess1(e,rows_number1,columns_number1) #Data Processing
	testData = DataProcess2(e) #Data Processing

	#training_labels preprocessing:
	fd3 = open(training_labels,"rb")
	image_number = getInfoLabel(fd3)
	trainingLabels = createNParrayLabel(fd3,image_number)
	fd3.close()

	#test_labels preprocessing:
	fd4 = open(test_labels,"rb")
	image_number = getInfoLabel(fd4)
	testLabels = createNParrayLabel(fd4,image_number)
	fd4.close()


	train_Y_one_hot = to_categorical(trainingLabels)
	#if trainingLabels[0] is 8, then train_Y_one_hot[0] is vector [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
	test_Y_one_hot = to_categorical(testLabels)

	train_X,valid_X,train_label,valid_label = train_test_split(trainingData,train_Y_one_hot,test_size=0.2,random_state=13)
	#train_X,train_label,testData,test_Y_one_hot

	inChannel = 1
	x, y = rows_number, columns_number
	input_img = Input(shape = (x, y, inChannel)) #input of encoder

	autoencoder_model = keras.models.load_model(autoencoderModel) #load autoencoder
	encoderLayers = 3 + 2*ConvLayersEnc #encoder layers = InputLayer + 2*MaxpoolingLayers + ConvolutionalLayers + BatchNormalizationLayers
										#ConvolutionalLayers and BatchNormalizationLayers are equal
	encoded = Encoder(input_img, filtersPerLayer, ConvLayersEnc, x_filter, y_filter)

	listFiltersFC = []
	listEpochs = []
	listBatchSize = []
	listLoss = []
	listTestLoss = []
	listAccuracy = []
	listTestAccuracy = []
	listPredictedClasses = []

	while True:

		filtersFC = int(input("Give number of filters at fully connected layer: "))
		epochs = int(input("Give epochs: "))
		batch_size = int(input("Give batch size: "))

		full_model = Model(input_img,fullyConnected(encoded,filtersFC))

		for l1,l2 in zip(full_model.layers[:encoderLayers],autoencoder_model.layers[0:encoderLayers]):
			l1.set_weights(l2.get_weights()) #full_model takes encoder weights from autoencoder
		##############
		for layer in full_model.layers[0:encoderLayers]: #at first I do not train the encoder part of full model
			layer.trainable = False

		full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
		full_model.summary()

		classify_train = full_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
		##############
		for layer in full_model.layers[0:encoderLayers]: #at second time I will train the full model again with the encoder part
			layer.trainable = True

		full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

		classify_train = full_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
		#full_model.save('classification.h5')
		##############
		listFiltersFC.append(filtersFC)
		listEpochs.append(epochs)
		listBatchSize.append(batch_size)

		loss = classify_train.history['loss']
		listLoss.append(loss[epochs-1])
		accuracy = classify_train.history['accuracy']
		listAccuracy.append(accuracy[epochs-1])

		testEvaluate = full_model.evaluate(testData, test_Y_one_hot, verbose=0)
		listTestLoss.append(testEvaluate[0])
		listTestAccuracy.append(testEvaluate[1])

		predictedClasses = full_model.predict(testData)
		predictedClasses = np.argmax(np.round(predictedClasses),axis=1)
		listPredictedClasses.append(predictedClasses)

		print("Give 1 if you want to do another experiment\nGive 2 if you want plots\nGive 3 if you want to classify the images of test set")
		doNext = int(input("Choose 1,2 or 3: "))

		if doNext == 2:
			f, axs = plt.subplots(6,1,figsize=(25,42))
			##########
			xs = list(range(len(listEpochs)))
			int2str = []
			for value in listEpochs:
				int2str.append(str(value))

			plt.subplot(6, 1, 1)
			plt.xticks(xs,int2str)
			plt.plot(xs, listAccuracy, 'bo', label='training accuracy')
			plt.plot(xs, listTestAccuracy, 'gs', label='test accuracy')
			plt.xlabel('epochs at each experiment')
			plt.ylabel('acc/test_acc')
			plt.title('Accuracy/Test_Accuracy per number of epochs')
			plt.legend()

			plt.subplot(6, 1, 2)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listTestLoss, 'gs', label='test loss')
			plt.xlabel('epochs at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per number of epochs')
			plt.legend()

			xs.clear()
			int2str.clear()
			##########
			xs = list(range(len(listBatchSize)))
			int2str = []
			for value in listBatchSize:
				int2str.append(str(value))

			plt.subplot(6, 1, 3)
			plt.xticks(xs,int2str)
			plt.plot(xs, listAccuracy, 'bo', label='training accuracy')
			plt.plot(xs, listTestAccuracy, 'gs', label='test accuracy')
			plt.xlabel('batch_size at each experiment')
			plt.ylabel('acc/test_acc')
			plt.title('Accuracy/Test_Accuracy per batch_size')
			plt.legend()

			plt.subplot(6, 1, 4)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listTestLoss, 'gs', label='test loss')
			plt.xlabel('batch_size at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per batch_size')
			plt.legend()

			xs.clear()
			int2str.clear()
			##########
			xs = list(range(len(listFiltersFC)))
			int2str = []
			for value in listFiltersFC:
				int2str.append(str(value))

			plt.subplot(6, 1, 5)
			plt.xticks(xs,int2str)
			plt.plot(xs, listAccuracy, 'bo', label='training accuracy')
			plt.plot(xs, listTestAccuracy, 'gs', label='test accuracy')
			plt.xlabel('filters of FC at each experiment')
			plt.ylabel('acc/test_acc')
			plt.title('Accuracy/Test_Accuracy per number of filters at FC layer')
			plt.legend()

			plt.subplot(6, 1, 6)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listTestLoss, 'gs', label='test loss')
			plt.xlabel('filters of FC at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per number of filters at FC layer')
			plt.legend()

			xs.clear()
			int2str.clear()
			##########
			plt.show()
			plt.savefig("classifyPlots.png")

			#Let's print arrays for precision,recall,f1-score,support (of previously experiments):
			for predictedClass in listPredictedClasses:
				classesNames = ["Class {}".format(i) for i in range(10)]
				print(classification_report(testLabels, predictedClass, target_names=classesNames))

			break
		if doNext == 3:
			#Correct Labels:
			correct = np.where(predictedClasses==testLabels)[0]
			f, axs = plt.subplots(4,4,figsize=(25,15))
			print("Found %d correct labels" % len(correct))
			for i, correct in enumerate(correct[:16]): #print 16 correct labels
				plt.subplot(4,4,i+1)
				plt.imshow(testData[correct].reshape(28,28), cmap='gray', interpolation='none')
				plt.title("Predicted {}, Class {}".format(predictedClasses[correct], testLabels[correct]))
				plt.tight_layout()
				plt.savefig("correctFC.png")
			#Incorrect Labels:
			incorrect = np.where(predictedClasses!=testLabels)[0]
			print("Found %d incorrect labels" % len(incorrect))
			for i, incorrect in enumerate(incorrect[:16]): #print 16 incorrect labels
				plt.subplot(4,4,i+1)
				plt.imshow(testData[incorrect].reshape(28,28), cmap='gray', interpolation='none')
				plt.title("Predicted {}, Class {}".format(predictedClasses[incorrect], testLabels[incorrect]))
				plt.tight_layout()
				plt.savefig("incorrectFC.png")

			break


if __name__ == "__main__":
	main()
