
from tensorflow import keras
import argparse
import numpy as np
import cv2
import os
path = os.getcwd()
from image_processing import image_procesor

def image_to_digits(grid,model,args):
    grid_numbers=np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image=grid[i*9+j]
            if  min(map(min,image)) > 127:
                grid_numbers[i][j]=0
                continue

            #if args.debug:
                #cv2.imwrite(path+"/Debug/network_feed/grid_"+str(i)+str(j)+".png",image)

            image = cv2.resize(image, (28, 28),interpolation = cv2.INTER_AREA)
            image,_,_=image_procesor(image)

            if args.debug:
                cv2.imwrite(path+"/Debug/network_feed/grid_"+str(i)+str(j)+".png",image)
                        
            image=image.reshape(1,28,28,1)
            probabs = model.predict(image)
            if np.max(probabs)<0.8:
                grid_numbers[i][j]=0

            else:
                grid_numbers[i][j]= np.argmax(model.predict(image) )+1

    return grid_numbers
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-t","--train",action="store_true",help="Shows output at each step")
# args = parser.parse_args()
# if args.train:
#
