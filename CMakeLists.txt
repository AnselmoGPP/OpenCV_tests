CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(testing)

ADD_EXECUTABLE( ${PROJECT_NAME}
	src/main.cpp
	src/header.cpp
)

TARGET_INCLUDE_DIRECTORIES( ${PROJECT_NAME} PUBLIC
	include

	../opencv-4.1.1/include
	../opencv-4.1.1/_BUILD
	../opencv-4.1.1/modules/core/include
	../opencv-4.1.1/modules/calib3d/include
	../opencv-4.1.1/modules/features2d/include
	../opencv-4.1.1/modules/flann/include
	../opencv-4.1.1/modules/dnn/include
	../opencv-4.1.1/modules/highgui/include
	../opencv-4.1.1/modules/imgcodecs/include
	../opencv-4.1.1/modules/videoio/include
	../opencv-4.1.1/modules/imgproc/include
	../opencv-4.1.1/modules/ml/include
	../opencv-4.1.1/modules/objdetect/include
	../opencv-4.1.1/modules/photo/include
	../opencv-4.1.1/modules/stitching/include
	../opencv-4.1.1/modules/video/include

#	../opencv-4.1.1/modules/gapi/include
#	../opencv-4.1.1/modules/ts/include
#	../opencv-4.1.1/modules/world/include
#	../opencv-4.1.1/modules/java/include
#	../opencv-4.1.1/modules/js/include
#	../opencv-4.1.1/modules/python/include
)

FILE(GLOB CV_HEADERS
	../opencv-4.1.1/_BUILD/lib/*.a
)

message( STATUS "OCV headers: " ${CV_HEADERS} )

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} PUBLIC
	${CV_HEADERS}
)
