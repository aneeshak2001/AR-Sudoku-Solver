import argparse
import cv2
import os
import numpy as np
from tensorflow import keras
from image_processing import image_procesor,sudoku_finder,grid_cropper,write_numbers
from digit_recognition import image_to_digits
from sudoku_solver import solve
import sys
import shutil
import time
from Webcam_capture import get_frame

# adding the path of the file
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,help="Path of input sudoku file")
parser.add_argument("-d","--debug",action="store_true",help="Shows output at each step")
args = parser.parse_args()

#Loading Model
try:
    model = keras.models.load_model('Model/big_epoch.h5')
except:
    sys.exit(" -- Provided path has no model file to read -- ")

#Making directories needed for debugging
if args.debug:
    print("Debug mode is on")
    # detect the current working directory and print it
    path = os.getcwd()
    os.makedirs(path+"/Debug/Numbers",exist_ok=True)
    os.makedirs(path+"/Debug/network_feed",exist_ok=True)
    os.makedirs(path+"/Debug/processing",exist_ok=True)

start = time.time()

#Reading Input
if args.path:
    image_input=cv2.imread(args.path,0)
    if image_input is None:
        sys.exit(" -- Provided path has no image file to read -- ")
else:
    image_input=get_frame()
    if image_input is None:
        sys.exit(" -- Image Was not captured, Exiting... -- ")

#Saving a Copy of the Input
image_copy=image_input.copy()

#Image Processing
_,Processed_image_dialated,Processed_image=image_procesor(image_input)

if args.debug:
    cv2.imwrite(path+"Debug/Input.png",image_copy)
    cv2.imwrite(path+"/Debug/Processed_image_dialated.png",Processed_image_dialated)
    cv2.imwrite(path+"/Debug/Processed_image.png",Processed_image)

#Cropping the sudoku
Sudoku_image=sudoku_finder(Processed_image_dialated,Processed_image)


if args.debug:
        cv2.imwrite(path+"/Debug/sudoku_cropped.png",Sudoku_image)

#Cropping the individual numbers
Squares=grid_cropper(Sudoku_image) #also cleans the image and adds None for empty

if args.debug:
    for i in range(9):
        for j in range(9):
            cv2.imwrite(path+"/Debug/Numbers/grid_"+str(i)+str(j)+".png",Squares[i*9+j])

if args.debug:
    shutil.rmtree(path+"/Debug/network_feed",ignore_errors=True)
    os.makedirs(path+"/Debug/network_feed",exist_ok=True) 

#Converting from Images to Numbers using the imported DL Model
grid_numbers=image_to_digits(Squares,model,args)


if args.debug:

    print(np.matrix(grid_numbers))

#Solving the Sudoku
solved_sudoku=solve(grid_numbers)

end=time.time()

if args.debug:

    print(np.matrix(solved_sudoku))

#Making the solution image
solved_sudoku_image = write_numbers(Sudoku_image, solved_sudoku, grid_numbers)


if args.debug:
    print("TIME TAKEN TO PROCESS: ",end-start)
    cv2.imwrite(path+"/Debug/solved_sudoku.png",solved_sudoku_image)    


#cv2.imshow("Solved Sudoku", solved_sudoku_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()    
#print("time = ",end-start)