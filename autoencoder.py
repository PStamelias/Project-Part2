#exw mono ta data arxeia
import sys
from matplotlib import pyplot as plt
import numpy as np
#import gzip
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
                if pos == 0: #temp array has 4 integers, each integer represents a byte of number  number_of_images
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
        train_X,valid_X,train_ground,valid_ground = train_test_split(e,e,test_size = 0.2,
random_state = 13)
        return train_X,valid_X,train_ground,valid_ground

#Built Encoder of autoencoder
def Encoder(input_img, filtersPerLayer, ConvLayersEnc, x_filter, y_filter):

        count = 0

        for value in filtersPerLayer:
                if count == ConvLayersEnc:
                        break
                elif count == 0: #for first Conv Layer
                        x = Conv2D(value, (x_filter, y_filter), activation='relu',padding='same')(input_img)
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
                x = Conv2D(filtersPerLayer[value], (x_filter, y_filter), activation='relu',padding='same')(x)               #except the last layer
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


        numOfLayers = int(input("Give number of layers: "))
        x_filter = int(input("Give the x coordinate of filter: "))
        y_filter = int(input("Give the y coordinate of filter: "))

        filtersPerLayer = [] #list with number of filters at each layer
        for val in range(numOfLayers):
                flt = int(input("Give number of filters in {} layer: ".format(val+1)))
                filtersPerLayer.append(flt)

        epochs = int(input("Give epochs: "))
        batch_size = int(input("Give batch size: "))

        inChannel = 1
        x, y = rows_number, columns_number
        input_img = Input(shape = (x, y, inChannel)) #input of autoencoder

        if numOfLayers%2 == 0: #if numOfLayers = even number
                ConvLayersEnc = numOfLayers//2 #how many Conv Layers the encoder has
                ConvLayersDec = numOfLayers//2 #how many Conv Layers the decoder has
        elif numOfLayers%2 != 0: #if numOfLayers = odd number
                ConvLayersEnc = numOfLayers//2
                ConvLayersDec = (numOfLayers//2) + 1


        autoencoder = Model(input_img, Decoder(Encoder(input_img,filtersPerLayer,ConvLayersEnc, x_filter, y_filter),filtersPerLayer, ConvLayersEnc, ConvLayersDec,x_filter, y_filter))
        autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

        autoencoder.summary()

        autoencoder_train = autoencoder.fit(train_X, train_ground,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X,valid_ground))


        #loss = autoencoder_train.history['loss']
        #val_loss = autoencoder_train.history['val_loss']
        #myepochs = range(epochs)
        #plt.figure()
        #plt.plot(myepochs, loss, 'bo', label='Training loss')
        #plt.plot(myepochs, val_loss, 'b', label='Validation loss')
        #plt.title('Training and validation loss')
        #plt.legend()
        #plt.savefig("temp.png")



if __name__ == "__main__":
        main()
