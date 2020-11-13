import sys
from matplotlib import pyplot as plt
import numpy as np
import gzip
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop

#Function that converts Big-Endian number to Little-Endian number 
def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))

#Opening the file and getting useful Info from it
def fun1(file):
	f=open(file,"rb")
	a = np.fromfile(f,dtype=np.uint32)
	magic_number=swap32(a[0])
	img_number=swap32(a[1])
	rows_number=swap32(a[2])
	columns_number=swap32(a[3])
	f.close()
	return img_number,rows_number,columns_number
	

#Opening the file and Creating the NumPy Array
def fun2(file,image_number,rows_number,columns_number):
	f=open(file,"rb")
	f.seek(16)
	e=f.read(image_number*rows_number*columns_number)
	e=np.frombuffer(e,dtype=np.uint8).astype(np.int64)
	e = e.reshape(image_number, rows_number,columns_number)
	f.close()
	return e

#Converting each rows_number x columns_number image of train set into a matrix of size rows_number x columns_number x 1, which you can feed into the Neural network
def fun3(e,rows_number,columns_number):
	e=e.reshape(-1,rows_number,columns_number,1)
	return e

#Rescaling the training  data with the maximum pixel value of the training 
def fun4(e):
	e=e/np.max(e)
	return e


def fun5(e):
	train_X,valid_X,train_ground,valid_ground = train_test_split(e,e,test_size=0.2,random_state=13)
	return train_X,valid_X,train_ground,valid_ground


def main():
	file=sys.argv[2]#File Name 
	image_number,rows_number,columns_number=fun1(file)#Getting info about Dataset
	e=fun2(file,image_number,rows_number,columns_number)#Creating NumPy Array
	e=fun3(e,rows_number,columns_number)#Data Processing 
	e=fun4(e)#Data Processing
	train_X,valid_X,train_ground,valid_ground=fun5(e)

if __name__ == "__main__":
    main()