
#include "opencv2/opencv.hpp"

#include "header.hpp"
#include <iostream>

using namespace cv;

void show_image(){

    // Load image
    char imageName[] = "Ned-Stark.jpg";
    Mat image = imread(imageName);
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // Show image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", image);

    waitKey(0);
}
