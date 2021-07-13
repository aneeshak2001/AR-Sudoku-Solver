# AR-Sudoku-Solver
This is a sudoku solver that takes an image/video of a sudoku as input and outputs a solved sudoku overlayed onto the initial input. 

## Requirements
Python >3.5
Python Packages: Opencv, numpy, Tensorflow
## To run:
1. Download all files and place them in a folder.
2. Run the 'main.py' file with the command "python main.py -p Sample_Images/test.jpeg" ('Sample_Images/test.jpeg' can be replaced with any path of an image).
3. To debug the code add "-d" as a console parameter.
4. If "-p" console parameter is not specified, then image will be taken from the webcam feed.
## How To use:
1. Upon running 'main.py' without the console parameter "-p", a window pops up with the webcam feed.
2. Place the sudoku within the frame and press space to capture the image.
3. The Solved Sudoku would be shown as a popup window once the code finishes processing.

