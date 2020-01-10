
#include "opencv2/opencv.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "header.hpp"
#include <iostream>
#include <vector>

void show_image(){

    // >> Load image
    char imageName[] = "Ned-Stark.jpg";
    cv::Mat image = cv::imread(imageName);      // If you need grayscale image:  imread(filename, 0);
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // >> Show image
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);

    cv::waitKey(0);
    
    // close the window
    cv::destroyWindow("Original");		// cv::destroyAllWindows();
    image.release();
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
    cv::destroyWindow("Original");		// cv::destroyAllWindows();
    image.release();
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
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Draw histogram frame
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

    // Normalize result to [0, histImage.rows] (so we don't get very high values)
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Draw the line for each channel
    for(int i = 1; i < histSize; i++)
    {
        cv::line(histImage, 
                 cv::Point(bin_w*(i-1), hist_h-cvRound(b_hist.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h-cvRound(b_hist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage, 
                 cv::Point(bin_w*(i-1), hist_h-cvRound(g_hist.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h-cvRound(g_hist.at<float>(i))),
                 cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImage, 
                 cv::Point(bin_w*(i-1), hist_h-cvRound(r_hist.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h-cvRound(r_hist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // Show image
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);
    cv::namedWindow("Histogram", cv::WINDOW_AUTOSIZE);
    cv::imshow("Histogram", histImage);

    cv::waitKey(0);
    cv::destroyAllWindows();
    image.release();
    histImage.release();
}

void get_LUT(){

    // Load image
    cv::Mat image = cv::imread("Ned-Stark.jpg");
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // Matrix 1x256 of 8U values
    cv::Mat lut(1, 256, CV_8U);

    // Transform the pixel values
    for(int i = 0; i < 256; i++)
    {
        //lut.at<uchar>(i) = 255 -i;				    // Inverse function
        lut.at<uchar>(i) = pow((float)i * 255, (float)(1/2.));      // Square root function
        //lut.at<uchar>(i) = pow((float)i, (float)2.) / (255);      // Square function
        //lut.at<uchar>(i) = pow((float)i, (float)3.) / (255*255);  // Cubic function
    }

    // Transform the image using the LUT
    cv::LUT(image, lut, image);

    // Show image
    namedWindow("Original", cv::WINDOW_AUTOSIZE);
    imshow("Original", image);

    cv::waitKey(0);
    cv::destroyWindow("Original");		// cv::destroyAllWindows();
    image.release();
}

void color_space(){

    // Load image
    cv::Mat imBGR = cv::imread("Ned-Stark.jpg");
    if(!imBGR.data) { std::cout << "Cannot load image" << std::endl; return; }

    // Split image into 3 sub-images (B,G,R)
    std::vector<cv::Mat> bgr_planes;
    cv::split(imBGR, bgr_planes);

    // Convert image from RGB to HSV
    cv::Mat imHSV;
    cv::cvtColor(imBGR, imHSV, cv::COLOR_BGR2HSV);

    // Split image into 3 sub-images (H,S,V)
    std::vector<cv::Mat> hsv_planes;
    cv::split(imHSV, hsv_planes);

    // Show RGB image and its components
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", imBGR);
    cv::namedWindow("Blue", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blue", bgr_planes[0]);
    cv::namedWindow("Green", cv::WINDOW_AUTOSIZE);
    cv::imshow("Green", bgr_planes[1]);
    cv::namedWindow("Red", cv::WINDOW_AUTOSIZE);
    cv::imshow("Red", bgr_planes[2]);

    // Show HSV image and its components
    cv::namedWindow("HSV", cv::WINDOW_AUTOSIZE);
    cv::imshow("HSV", imHSV);
    cv::namedWindow("Hue", cv::WINDOW_AUTOSIZE);
    cv::imshow("Hue", hsv_planes[0]);
    cv::namedWindow("Saturation", cv::WINDOW_AUTOSIZE);
    cv::imshow("Saturation", hsv_planes[1]);
    cv::namedWindow("Value", cv::WINDOW_AUTOSIZE);
    cv::imshow("Value", hsv_planes[2]);

    cv::waitKey(0);
    cv::destroyAllWindows();
    imBGR.release();
    bgr_planes[0].release();
    bgr_planes[1].release();
    bgr_planes[2].release();
    imHSV.release();
    hsv_planes[0].release();
    hsv_planes[1].release();
    hsv_planes[2].release();
}

void arithmetic(){

    // Load image
    cv::Mat image = cv::imread("Ned-Stark.jpg");
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // Arithmetic operations
    cv::Mat result;
    cv::add(image, image, result, cv::noArray(), -1);
    cv::imshow("ADD", result);
    cv::subtract(image, image, result, cv::noArray(), -1);
    cv::imshow("SUBTRACT", result);
    cv::multiply(image, image, result, (1.0), -1);
    cv::imshow("MULTIPLY", result);
    cv::divide(image, image, result, (1.0), -1);
    cv::imshow("DIVIDE", result);

    cv::waitKey(0);
    cv::destroyAllWindows();
    result.release();
}

void logic(){

    // Create images
    cv::Mat img1 = cv::Mat::zeros(cv::Size(400, 200), CV_8UC1);         // Size(X, Y)
    cv::Mat img2 = cv::Mat::zeros(cv::Size(400, 200), CV_8UC1);

    img1(cv::Range(0, img1.rows), cv::Range(0, img1.cols/2)) = 255;     // img1(Range(Y), Range(X))
    cv::imshow("img1", img1);
    img2(cv::Range(100, 150), cv::Range(150, 350)) = 255;
    cv::imshow("img2", img2);

    cv::Mat res;
    cv::bitwise_and(img1, img2, res);
    cv::imshow("AND", res);
    cv::bitwise_or(img1, img2, res);
    cv::imshow("OR", res);
    cv::bitwise_xor(img1, img2, res);
    cv::imshow("XOR", res);
    cv::bitwise_not(img1, res);
    imshow("NOT", res);

    cv::waitKey(0);
    cv::destroyAllWindows();
    res.release();
}

void make_mask(){

    // Load image
    cv::Mat image = cv::imread("Ned-Stark.jpg");
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    // Generate a binary image (mask)
    for(int x = 0; x < image.cols; ++x)
        for(int y = 0; y < image.rows; ++y)
        {
            if( (image.at<cv::Vec3b>(cv::Point(x,y))[0] + image.at<cv::Vec3b>(cv::Point(x,y))[1] +
		 image.at<cv::Vec3b>(cv::Point(x,y))[2]) / 3 < 255/6 )
            {
 	        image.at<cv::Vec3b>(cv::Point(x,y)) = cv::Vec3b(0, 0, 0);
            }
            else image.at<cv::Vec3b>(cv::Point(x,y)) = cv::Vec3b(255, 255, 255);
        }

    cv::imshow("Original masked", image);
    cv::imwrite("Ned_Stark_mask_2.jpg", image);

    cv::waitKey(0);
    cv::destroyWindow("Original masked");		// cv::destroyAllWindows();
    image.release();
}

void draw(){
    
    // Load image
    cv::Mat image = cv::imread("Ned-Stark.jpg");
    if(!image.data) { std::cout << "Cannot load image" << std::endl; return; }

    std::cout << "Image size: " << image.cols << ", " << image.rows << std::endl;

    // Draw (line, rectangle, polyline, text)
    cv::line(image, cv::Point(0, 0), cv::Point(600, 350), cv::Scalar(255, 255, 255), 10);

    cv::rectangle(image, cv::Point(600, 350), cv::Point(700, 500), cv::Scalar(255, 0, 0), 5);	// -1 to fill the shape

    cv::circle(image, cv::Point(650, 425), 20, cv::Scalar(0, 255, 0), 5);			// -1 to fill the shape

    std::vector<cv::Point> pts = { cv::Point(650, 425), cv::Point(800, 350), cv::Point(850, 600) };
    cv::polylines(image, pts, true, cv::Scalar(255, 255, 255), 5);	// true (closes polygon), false (doesn't close it)

    cv::putText(image, "Hello, image", cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 3.5, cv::Scalar(0, 100, 200), 5, cv::LINE_AA);	// LINE_AA gives some antialiasing 
    
    cv::imshow("Original", image);

    cv::waitKey(0);
    cv::destroyWindow("Original");
    image.release();
}



