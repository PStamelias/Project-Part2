import sys
from matplotlib import pyplot as plt
import numpy as np
import gzip
import numpy as np
import pandas as pd
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop

def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))

def fun1(file):
	f=open(file,"r")
	a = np.fromfile(f,dtype=np.uint32)
	magic_number=swap32(a[0])
	img_number=swap32(a[1])
	rows_number=swap32(a[2])
	columns_number=swap32(a[3])
	f.close()
	
	

def main():
	file=sys.argv[2]
	fun1(file)
	

if __name__ == "__main__":
    main()