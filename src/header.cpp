
#include "opencv2/opencv.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "header.hpp"
#include <iostream>

void show_image(){

    // >> Load image
    char imageName[] = "Ned-Stark.jpg";
    cv::Mat image = cv::imread(imageName);      // If you need grayscale image:  imread(filename, 0);
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // >> Show image
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);

    cv::waitKey(0);
}

void print_rectangle(){

    // Load image
    cv::Mat image = cv::imread("Ned-Stark.jpg");
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // >> Print rectangle
    for(int x = image.cols / 4; x < 3 * image.cols / 4; ++x)
        for(int y = image.rows / 4; y < 3 * image.rows / 4; ++y)
            image.at<cv::Vec3b>(cv::Point(x,y)) = cv::Vec3b(0, 0, 0);

    // Show image
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);

    // >> Save result
    cv::imwrite("Rectangle.jpg", image);

    cv::waitKey(0);
}

void get_histogram(){

    // Load image
    cv::Mat image = cv::imread("Ned-Stark.jpg");
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // Separate the image in 3 sub-images (B,G,R)
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes);

    // Histogram variables
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = { range };
    bool uniform = true;               // Uniform histogram
    bool accumulate = false;           // Non-accumulated histogram
    cv::Mat b_hist, g_hist, r_hist;
    
    // Calculate histogram
    calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, $histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, $histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, $histRange, uniform, accumulate);









    // Show image
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);

    // >> Save result
    cv::imwrite("Rectangle.jpg", image);
}

