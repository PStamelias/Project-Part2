#exw mono ta data arxeia
import sys
from matplotlib import pyplot as plt
import numpy as np
import six
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop


#Getting useful Info from file with file descriptor f
def getInfo(f):
	temp = f.read(4) #pass first 4 bytes that are the magic number

	for pos in range(3):
		temp = f.read(4) #read 4 bytes
		temp = np.frombuffer(temp,dtype = np.uint8).astype(np.int64) #convert the string that stored in temp into an array(numpy) of integers
		if pos == 0: #temp array has 4 integers, each integer represents a byte of number number_of_images
			img_number = (temp[0]<<24)|(temp[1]<<16)|(temp[2]<<8)|temp[3] #convert the temp array into an integer
		elif pos == 1: #temp has number of rows
			rows_number = (temp[0]<<24)|(temp[1]<<16)|(temp[2]<<8)|temp[3]
		elif pos == 2: #temp has number of columns
			columns_number = (temp[0]<<24)|(temp[1]<<16)|(temp[2]<<8)|temp[3]

	return img_number,rows_number,columns_number


#Creating NumPy Array with images from file with file descriptor f
def createNParray(f,image_number,rows_number,columns_number):
	e = f.read(image_number*rows_number*columns_number)     #at the beginning e is string with pixels of image_number images
	e = np.frombuffer(e,dtype = np.uint8).astype(np.int64)  #convert string to numpy array of integers
	e = e.reshape(image_number, rows_number,columns_number) #convert the numpy array into three-dimensional array with dimensions
                                              				#image_numberxrows_numberxcolumns_number
	return e

#Converting each rows_number x columns_number image of train set into a matrix of size rows_number x columns_number x 1, which you can feed into the Neural network
def DataProcess1(e,rows_number,columns_number):
	e = e.reshape(-1,rows_number,columns_number,1) #convert every image rows_numberxcolumns_number to vector rows_numberxcolumns_numberx1
	return e

#Rescaling the training  data with the maximum pixel value of the training
def DataProcess2(e): #convert every pixel to 0 or 1
	e = e/np.max(e)
	return e

#Split the dataset into training and validation set (this reduce the probability of overfitting)
def DataProcess3(e):
	train_X,valid_X,train_ground,valid_ground = train_test_split(e,e,test_size = 0.2, random_state = 13)
	return train_X,valid_X,train_ground,valid_ground

#user gives hyperparameters
def GiveHyperparameters():

	numOfLayers = int(input("Give number of layers: "))
	x_filter = int(input("Give the x coordinate of filter: "))
	y_filter = int(input("Give the y coordinate of filter: "))

	filtersPerLayer = [] #list with number of filters at each layer
	for val in range(numOfLayers):
		flt = int(input("Give number of filters in {} layer: ".format(val+1)))
		filtersPerLayer.append(flt)

	epochs = int(input("Give epochs: "))
	batch_size = int(input("Give batch size: "))

	if numOfLayers%2 == 0: #if numOfLayers = even number
		ConvLayersEnc = numOfLayers//2 #how many Conv Layers the encoder has
		ConvLayersDec = numOfLayers//2 #how many Conv Layers the decoder has
	elif numOfLayers%2 != 0: #if numOfLayers = odd number
		ConvLayersEnc = numOfLayers//2
		ConvLayersDec = (numOfLayers//2) + 1

	return numOfLayers,x_filter,y_filter,filtersPerLayer,epochs,batch_size,ConvLayersEnc,ConvLayersDec


#Built Encoder of autoencoder
def Encoder(input_img, filtersPerLayer, ConvLayersEnc, x_filter, y_filter):
	count = 0

	for value in filtersPerLayer:
		if count == ConvLayersEnc:
			break
		elif count == 0: #for first Conv Layer
			x = Conv2D(value, (x_filter, y_filter), activation='relu', padding='same')(input_img)
			x = BatchNormalization()(x)
			x = MaxPooling2D(pool_size=(2, 2))(x) #first MaxPooling
		elif count != 0:
			x = Conv2D(value, (x_filter, y_filter), activation='relu', padding='same')(x)
			x = BatchNormalization()(x)
			if count == 1:
				x = MaxPooling2D(pool_size=(2, 2))(x) #second MaxPooling (we have 2 MaxPooling Layers)
		count = count + 1

	return x

#Built Decoder of autoencoder
def Decoder(x, filtersPerLayer, ConvLayersEnc, ConvLayersDec, x_filter, y_filter):

	layerUpSampl1 = ConvLayersEnc + ConvLayersDec - 2
	layerUpSampl2 = ConvLayersEnc + ConvLayersDec - 3

	for value in range(ConvLayersEnc, ConvLayersEnc + ConvLayersDec - 1): #take the number of filters which belong in decoder convolutional layers
		x = Conv2D(filtersPerLayer[value], (x_filter, y_filter), activation='relu', padding='same')(x)               #except the last layer
		x = BatchNormalization()(x)
		if value == layerUpSampl1 or value == layerUpSampl2: #layers followed by UpSampling layers
			x = UpSampling2D((2,2))(x)

	decoded = Conv2D(1, (x_filter, y_filter), activation='sigmoid', padding='same')(x)
	return decoded


def main():
	for pos in range(len(sys.argv)):
		if sys.argv[pos] == '-d':  #if the current argument is '-d' then
			file = sys.argv[pos+1] #the next argument has the file name of dataset
			break

	fd = open(file,"rb")
	image_number,rows_number,columns_number = getInfo(fd) #Getting info about Dataset
	e = createNParray(fd,image_number,rows_number,columns_number) #Creating NumPy Array
	fd.close()

	e = DataProcess1(e,rows_number,columns_number) #Data Processing
	e = DataProcess2(e) #Data Processing
	train_X,valid_X,train_ground,valid_ground = DataProcess3(e)


	inChannel = 1
	x, y = rows_number, columns_number
	input_img = Input(shape = (x, y, inChannel)) #input of autoencoder

	listNumOfLayers = []
	list_xfilter = []
	list_yfilter = []
	listFiltersPerLayer = [] #list of lists,that list consists of filtersPerLayer lists
	listEpochs = []
	listBatchSize = []
	listLoss = []
	listValLoss = []


	while True:

		numOfLayers,x_filter,y_filter,filtersPerLayer,epochs,batch_size,ConvLayersEnc,ConvLayersDec = GiveHyperparameters()

		autoencoder = Model(input_img, Decoder(Encoder(input_img,filtersPerLayer, ConvLayersEnc, x_filter, y_filter),filtersPerLayer, ConvLayersEnc, ConvLayersDec, x_filter, y_filter))
		autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

		autoencoder.summary()

		#autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

		listNumOfLayers.append(numOfLayers)
		list_xfilter.append(x_filter)
		list_yfilter.append(y_filter)
		listFiltersPerLayer.append(filtersPerLayer)
		listEpochs.append(epochs)
		listBatchSize.append(batch_size)

		#loss = autoencoder_train.history['loss']
		#val_loss = autoencoder_train.history['val_loss']
		#listLoss.append(loss[epochs-1]) #add at list loss value from last epoch
		#listValLoss.append(val_loss[epochs-1]) #add at list val_loss value from last epoch
		listLoss = []
		listValLoss = []

		print("Give 1 if you want to do another experiment\nGive 2 if you want plots\nGive 3 if you want to save the current model")
		doNext = int(input("Choose 1,2 or 3: "))


		if doNext == 2:

			transpose = list(map(list,six.moves.zip_longest(*listFiltersPerLayer,fillvalue = -1)))
			#if listFiltersPerLayer = [[32,64,64,1],[56,40,4,12,1],[27,6,1]]
			#then transpose = [[32,56,27],[64,40,6],[64,4,1],[1,12,-1],[-1,1,-1]]
			#first experiment -> layer_1 has 32 filters, layer_2 has 64 filters ...
			#second experiment -> layer_1 has 56 filters, layer_2 has 40 filters...
			numplots = len(transpose) + 5
			#plt.figure()
			f, axs = plt.subplots(numplots,1,figsize=(25,numplots*7))

			xs = list(range(len(listEpochs)))
			int2str = []
			for value in listEpochs:
				int2str.append(str(value))

			plt.subplot(numplots, 1, 1)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listValLoss, 'gs', label='validation loss')
			plt.xlabel('epochs at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per number of epochs')
			plt.legend()

			xs.clear()
			int2str.clear()

			xs = list(range(len(listBatchSize)))
			for value in listBatchSize:
				int2str.append(str(value))

			plt.subplot(numplots, 1, 2)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listValLoss, 'gs', label='validation loss')
			plt.xlabel('batch_size at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per batch_size')
			plt.legend()

			xs.clear()
			int2str.clear()

			xs = list(range(len(listNumOfLayers)))
			for value in listNumOfLayers:
				int2str.append(str(value))

			plt.subplot(numplots, 1, 3)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listValLoss, 'gs', label='validation loss')
			plt.xlabel('number of layers at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per num_of_layers')
			plt.legend()

			xs.clear()
			int2str.clear()

			xs = list(range(len(list_xfilter)))
			for value in list_xfilter:
				int2str.append(str(value))

			plt.subplot(numplots, 1, 4)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listValLoss, 'gs', label='validation loss')
			plt.xlabel('x coordinate of filter at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per filter size (x coordinate)')
			plt.legend()

			xs.clear()
			int2str.clear()

			xs = list(range(len(list_yfilter)))
			for value in list_yfilter:
				int2str.append(str(value))

			plt.subplot(numplots, 1, 5)
			plt.xticks(xs,int2str)
			plt.plot(xs, listLoss, 'bo', label='training loss')
			plt.plot(xs, listValLoss, 'gs', label='validation loss')
			plt.xlabel('y coordinate of filter at each experiment')
			plt.ylabel('loss/val_loss')
			plt.title('Loss/Val_Loss per filter size (y coordinate)')
			plt.legend()

			xs.clear()
			int2str.clear()

			cnt = 6
			templistflt = []
			templistLoss = []
			templistValLoss = []
			for value in range(len(transpose)):
				temp1 = value + 1

				cnt1 = 0
				for pos in transpose[value]:
					if pos != -1:
						templistflt.append(transpose[value][cnt1])
						templistLoss.append(listLoss[cnt1])
						templistValLoss.append(listValLoss[cnt1])
					cnt1 = cnt1 + 1

				xs = list(range(len(templistflt)))
				for fltvalue in templistflt:
					int2str.append(str(fltvalue))

				plt.subplot(numplots, 1, cnt)
				plt.xticks(xs,int2str)
				plt.plot(xs, templistLoss, 'bo', label='training loss')
				plt.plot(xs, templistValLoss, 'gs', label='validation loss')
				plt.xlabel('number of filters')
				plt.ylabel('loss/val_loss')
				plt.title('Loss/Val_Loss per number_of_filters at layer %d' %temp1)
				plt.legend()

				xs.clear()
				int2str.clear()

				templistflt.clear()  #templistflt = []
				templistLoss.clear()
				templistValLoss.clear()
				cnt = cnt + 1


			plt.show()
			plt.savefig("temp.png")
			break
		if doNext == 3:
			path = input("Give the path: ")
			f = open("info.txt", "w")
			f.write(str(numOfLayers));
			f.write("\n");
			f.write(str(x_filter));
			f.write("\n");
			f.write(str(y_filter));
			f.write("\n");
			f.write(str(filtersPerLayer));
			f.write("\n");
			f.write(str(epochs));
			f.write("\n");
			f.write(str(batch_size));
			f.write("\n");
			f.write(str(ConvLayersEnc));
			f.write("\n");
			f.write(str(ConvLayersDec));
			f.write("\n");
			f.close()
			#autoencoder.save(path)
			print(numOfLayers,"\n",x_filter,"\n",y_filter,"\n",filtersPerLayer,"\n",epochs,"\n",batch_size,"\n",ConvLayersEnc,"\n",ConvLayersDec)
			break


	#ta numOfLayers,x_filter,y_filter,filtersPerLayer,epochs,batch_size,ConvLayersEnc,ConvLayersDec
	#exoyn tis times toy teleytaioy peiramatos


if __name__ == "__main__":
	main()
