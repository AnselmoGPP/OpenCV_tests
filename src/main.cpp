/*
	Notes about OpenCV:
		cv::waitKey(0) stops the execution and waits for user input to continue
		If the function where an image is opened in a window is finished without closing the window, it will remain and you won't be able to close it normally.
		If the program where an image is opened in a window is finished, the window will be closed.
*/


#include "header.hpp"
#include <iostream>

int main()
{
    int val;

    std::cout << '\n'
              << "0  - Exit \n"
              << "1  - Show image \n"
              << "2  - Print rectangle \n"
              << "3  - Get histogram \n"
              << "4  - Get LUT \n"
              << "5  - Color space \n"
              << "6  - Arithmetic \n"
              << "7  - Logic \n"
              << "8  - Make mask \n"
              << "9  - Draw \n"
              << std::endl;

    do{
        std::cout << "Select function: ";
        std::cin >> val;
        switch(val)
        {
            case 0:
            break;
            case 1:
            show_image();
            break;
            case 2:
            print_rectangle();
            break;
            case 3:
            get_histogram();
            break;
            case 4:
            get_LUT();
            break;
            case 5:
            color_space();
            break;
            case 6:
            arithmetic();
            break;
            case 7:
            logic();
            break;
            case 8:
            make_mask();
            break;
            case 9:
            draw();
            break;
            default:
            std::cout << "Not valid option \n" << std::endl;
            break;
        }
    }while(val != 0);
}

