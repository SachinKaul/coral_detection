# coral_detection
Image processing software which takes in an image of newborn coral and detects its area. 

The coral images are in the dataset folder as JPG files. These files are taken and cut into ppm files for image processing. 
The main C file uses the netpbm library and performs its image manipulation through ppm files. 
This code takes in one of the sample ppm inputs with a coral polyp within it and performs the following transformations.
First it smoothes the image using an averaging filter, then applies Canny Edge Detection to isolate the edges of the file, enhances those edges so that only intensity values of 0 or 255 remain. 
Then, since the newborn coral formations are circular in nature, a circular hough transformation is used to detect where they are. This detection is used as a mask to remove the noise elsewhere in the image. 
The final result contains the edges of only the detected newborn corals. This can be used to calculate the area of the newborn corals as well as to determine their location. 

This project was completed in collaboration with Harun Saib using the netpbm library as published by Marc Pomplun. 
