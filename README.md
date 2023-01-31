# blocs_matching_101


Block Matching Video Sequence Analysis

This code analyzes a video sequence using the block matching technique.

How it works

The frame is divided into 16x16 blocks

The similar block is searched for in a neighborhood (kxk) and the residual is calculated

Similar blocks are framed with squares of the same color (for visualizing results)

The residual image is displayed. If the block has not changed, a black rectangle will be displayed.

The search is optimized by applying binary search. The reference document will be available for consultation

Requirements

   OpenCV (cv2)
   
   Numpy
   
   math (inf)
   
Usage

 Run the code on the terminal
 
 Enter the path of the video sequence
 
 The results will be displayed in a window.
 
Output

   The original frame
   
   The matching frame
   
   The residual image
   
   The reconstructed image
   
