//
//  main.cpp
//  OpenCV_
//
//  Created by Magen on 8/3/23.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//// Chapter 1 - Read Images, Videos, and Webcam
//// Importing Images
////
//int main() {
//    string path = "/Users/magen/Desktop/test.png";
//    Mat img = imread(path); // will read image from the path
//    // mat is a matric data type that belongs to openCV and it
//    // will read all the images
//
//    // Display the image in the "Matches" window
//    imshow("Image", img);
//
//    waitKey(0); // will show but will close automatically
//    // we do 0 for it to show infinitely
//
//    return 0;
//}

// importing video

// videos are a series of images
// iterate through all the images and capture them one by one and display them using a while loop
//int main() {
//
//    string path = "Resources/test_video.mp4";
//    VideoCapture cap(path); // initialized with a valid path to a video file or a camera index
//    Mat img;
//
//    while(true) {
//        // Mat image = imread(path);
//        cap.read(img);
//        imshow("Image", img);
//        waitKey(0); // speed of mp4
//    }
//}

// webcam


int main() {
    // Create a VideoCapture object to access the webcam (device 0)
    VideoCapture cap(0);

    // Check if the VideoCapture object is successfully opened
    if (!cap.isOpened()) {
        cout << "Error: Unable to access the webcam." << endl;
        return -1;
    }

    // Create a Mat object to hold the webcam frame
    Mat img;

    // Infinite loop to continuously read and display frames from the webcam
    while (true) {
        // Read a frame from the webcam and store it in the 'img' Mat object
        cap.read(img);

        // Check if the frame is empty (indicating a problem with the webcam)
        if (img.empty()) {
            cout << "Error: Webcam frame is empty." << endl;
            break;
        }

        // Display the current frame from the webcam in a window titled "Webcam"
        imshow("Webcam", img);

        // Wait for 30 milliseconds (30ms) for a key press
        int key = waitKey(30);

        // Check if the user pressed 'q' (ASCII code 113)
        if (key == 'q')
            break;
    }

    // Close the window and release the webcam
    destroyAllWindows();

    return 0;
}

// Chapter 2 - Basic Functions
//
//int main() {
//    string path = "Resources/test.png";
//    Mat img = imread(path);
//
//    if (img.empty()) {
//        cout << "Error: Failed to read image!" << endl;
//        return -1;
//    }
//
//    Mat imgGray, imgBlur, imgCanny, imgDia, imgErode;
//    cvtColor(img, imgGray, COLOR_BGR2GRAY);
//  //  GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY)
//    GaussianBlur(imgGray, imgBlur, Size(7,7), 5,0);
//    Canny(imgBlur, imgCanny, 50, 150);
//    // 50 and 150 are threshold values
//    // low threshold - weak edges
//    // high threshold - strong edges
//    // black areas are blow the low and high thresholds
//
//    Mat kernal = getStructuringElement(MORPH_RECT, Size(5,5));
//    // lower decreases dilation, higher increases
//
//    dilate(imgCanny, imgDia, kernal); // increases thickness
//    // in python we use NUM_PI to define our kernal, since 3rd argument takes a kernal
//    erode(imgDia, imgErode, kernal); // decreases thickness
//
//    imshow("Image", img);
//    imshow("Image Gray", imgGray);
//    imshow("Image Blur", imgBlur);
//    imshow("Image Canny", imgCanny);
//    imshow("Image Dilation", imgDia);
//    imshow("Image Erode", imgErode);
//
//    waitKey(0);
//
//    return 0;
//}

// Chapter 3 - Resize & Crop
//int main() {
//    string path = "Resources/test.png";
//    Mat img = imread(path);
//    Mat imgResize, imgCrop;
//
//    cout << img.size() << endl;
//   // resize(img, imgResize, Size(640,480));
//    resize(img, imgResize, Size(), 0.5, 0.5);
//    // scaled by 50%
//
//    Rect roi(200, 100, 300, 300);
//    // x: The x-coordinate of the top-left corner of the rectangle.
//    // y: The y-coordinate of the top-left corner of the rectangle.
//    // width: The width of the rectangle.
//    // height: The height of the rectangle.
//    imgCrop = img(roi);
//
//    imshow("Image", img);
//    imshow("Image Resize", imgResize);
//    imshow("Image crop", imgCrop);
//    waitKey(0);

//    return 0;
//}

// Chapter 4 - Drawing Shapes & Text

//int main(){
//    // Blank Image
//    Mat img(512, 512, CV_8UC3, Scalar(255,255,255));
//    circle(img, Point(256,256), 100, Scalar(0,69,255), FILLED);
//    // void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0);
//    // 255 radius
//    // 5th parameter is the thickness ie. 10 or FILLED
//    rectangle(img, Point(130,226), Point(382,286), Scalar(255,255,255), FILLED);
//    // void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0);
//
//    // thickness is 3
//    // pt1: The top-left corner of the rectangle, specified as Point(x1, y1), where x1 and y1 are the coordinates of the top-left corner.
//    // pt2: The bottom-right corner of the rectangle, specified as Point(x2, y2), where x2 and y2 are the coordinates of the bottom-right corner.
//
//    line(img, Point(130,296), Point(382,296), Scalar(255,255,255));
//    // the point is starting and ending point
//    // line(img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2, LINE_AA);
//
//    putText(img, "hi there", Point(137,262), FONT_HERSHEY_SIMPLEX, 2, Scalar(255,0,0),2);
//    //void putText(Mat& img, const String& text, Point org, int fontFace, double fontScale, Scalar color, int thickness = 1, int lineType = LINE_8, bool bottomLeftOrigin = false);
//
//
//    /* unsigned is values 0 to 255
//    signed is values -127 to 127
//     512 height x 512 width dimensions
//    3 color channels for BGR
//    scalar defines our BGR values */
//    imshow("Image", img);
//    waitKey(0);
//    return 0;
//
//}

// Chapter 5 - Warp Perspective
//
//float w = 250, h = 350;
//Mat matrix, imgWarp;
//
//int main() {
//    // Replace "your_username" with your actual username
//    string desktopPath = "/Users/magen/Desktop/";
//
//    // Replace "cards.jpg" with the name of your image file
//    string imagePath = desktopPath + "cards.jpg";
//
//    Mat img = imread(imagePath);
//    if (img.empty()) {
//        cout << "Error: Failed to read image!" << endl;
//        return -1;
//    }
//    // source points
//    Point2f src[4] = {{529,142},{771,190},{405,395},{674,457}};
//
//    /*
//     This code defines an array of four 2D points using the Point2f class from OpenCV. Each point represents a coordinate in a 2D plane, specified by its x and y coordinates.
//
//     Here's what each point represents:
//
//     src[0]: (x=529, y=142)
//     src[1]: (x=771, y=190)
//     src[2]: (x=405, y=395)
//     src[3]: (x=674, y=457)
//     */
//
//    // destination points
//    Point2f dst[4] = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
//
//    /*
//     {0.0f, 0.0f}: This point represents the top-left corner of the destination rectangle.
//
//     {w, 0.0f}: This point represents the top-right corner of the destination rectangle. The value of w corresponds to the width of the destination rectangle.
//
//     {0.0f, h}: This point represents the bottom-left corner of the destination rectangle. The value of h corresponds to the height of the destination rectangle.
//
//     {w, h}: This point represents the bottom-right corner of the destination rectangle.
//     */
//
//    matrix = getPerspectiveTransform(src, dst);
//    warpPerspective(img, imgWarp, matrix, Point(w,h));
//
//    /* matrix = getPerspectiveTransform(src, dst);
//
//     getPerspectiveTransform() is a function from OpenCV that calculates the 3x3 perspective transformation matrix based on the source and destination points.
//     This matrix defines how the original quadrilateral region should be transformed to the rectangular region.
//
//     warpPerspective(img, imgWarp, matrix, Point(w,h));
//
//     warpPerspective() is another OpenCV function that applies the perspective transformation to the original image (img) using the transformation matrix (matrix) calculated earlier.
//     The result of the transformation is stored in the imgWarp variable.
//     Point(w, h) specifies the size of the output image, which is the width w and height h of the destination rectangle.
//     */
//
//    for(int i = 0; i < 4; i++) {
//        circle(img, src[i], 10, Scalar(0,0,255), FILLED);
//    }
//
//
//    imshow("Image", img);
//    imshow("Image Warp", imgWarp);
//    waitKey(0);
//
//    return 0;
//}

// Chapter 6 - Color Detection

//
//Mat imgHSV, mask;
//int hmin = 0, smin = 110, vmin = 153;
//int hmax = 19, smax = 240, vmax = 255;
//
//int main() {
//    // Replace "your_username" with your actual username
//    string desktopPath = "/Users/magen/Desktop/";
//
//    // Replace "cards.jpg" with the name of your image file
//    string imagePath = desktopPath + "shapes.png";
//
//    Mat img = imread(imagePath);
//    if (img.empty()) {
//        cout << "Error: Failed to read image!" << endl;
//        return -1;
//    }
//    cvtColor(img, imgHSV, COLOR_BGR2HSV);
//
//    namedWindow("Trackbars", WINDOW_OPENGL);
//    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
//    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
//    createTrackbar("Sat Min", "Trackbars", &smin, 255);
//    createTrackbar("Sat Max", "Trackbars", &smax, 255);
//    createTrackbar("Val Min", "Trackbars", &vmin, 255);
//    createTrackbar("Val Max", "Trackbars", &vmax, 255);
//
//    // void createTrackbar(const string& trackbarName, const string& windowName,
//    // int* value, int count, TrackbarCallback onChange = 0,
//    // void* userdata = 0);
//
//
//    // Resize the "Trackbars" window to the desired size (width x height)
//       resizeWindow("Trackbars", 640, 200);
//
//    // hue is 180, sat & value are 255 for max values
//
//    /*
//     The createTrackbar function is a feature provided by OpenCV to create a graphical slider (trackbar) on a window. Trackbars are typically used to interactively adjust parameters in real-time while viewing an image or video. This function allows you to create a trackbar associated with a specific window and link it to a variable in your code.
//
//     The syntax for createTrackbar is as follows:
//     void createTrackbar(const string& trackbarName, const string& windowName,
//                         int* value, int count, TrackbarCallback onChange = 0,
//                         void* userdata = 0);
//
//     */
//
//    while(true) {
//        Scalar lower(hmin, smin, vmin);
//        Scalar upper(hmax, smax, vmax);
//        inRange(imgHSV, lower, upper, mask);
//        // output is our mask
//
//        imshow("Image", img);
//        imshow("Image HSV",imgHSV);
//        imshow("Image Mask", mask);
//        waitKey(1);
//    }
//    return 0;
//}

// Chapter 7 - Shapes/Contour Detection
//
//Mat imgGray, imgBlur, imgCanny, imgDia, imgErode;
//void getContours(Mat imgDia, Mat img);
//
//int main() {
//    // Replace "your_username" with your actual username
//    string desktopPath = "/Users/magen/Desktop/";
//
//    // Replace "cards.jpg" with the name of your image file
//    string imagePath = desktopPath + "shapes.png";
//
//    Mat img = imread(imagePath);
//    if (img.empty()) {
//        cout << "Error: Failed to read image!" << endl;
//        return -1;
//    }
//
//    // preprocessing
//    cvtColor(img, imgGray, COLOR_BGR2GRAY);
//    GaussianBlur(imgGray, imgBlur, Size(3,3), 3, 0);
//    Canny(imgBlur, imgCanny, 25, 75);
//
//    Mat kernal = getStructuringElement(MORPH_RECT, Size(5,5));
//    // lower decreases dilation, higher increases
//
//    /* MORPH_RECT:
//
//     Represents a rectangular structuring element.
//     The most basic and commonly used structuring element.
//     Effective for general morphological operations like dilation and erosion.
//     It can expand or shrink regions with sharp edges.
//     MORPH_ELLIPSE:
//
//     Represents an elliptical structuring element.
//     Suitable for smoothing and noise reduction in an image.
//     More effective for rounding or smoothing out regions and contours.
//     Less sensitive to noise compared to MORPH_RECT.
//     MORPH_CROSS:
//
//     Represents a cross-shaped structuring element.
//     Often used for specific morphological operations.
//     It is useful for removing thin lines or small noise in a binary image.
//     Can be effective for tasks like skeletonization. */
//
//    dilate(imgCanny, imgDia, kernal); // increases thickness
//
//    getContours(imgDia, img);
//
//    imshow("Image", img);
////    imshow("Image Gray", imgGray);
////    imshow("Image Blur", imgBlur);
////    imshow("Image Canny", imgCanny);
////    imshow("Image Dil", imgDia);
//    waitKey(0);
//    return 0;
//}
//
//void getContours(Mat imgDia, Mat img)
//{
//    // each vector will be a contour and each contour will have some points
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//
//    findContours(imgDia, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//    // drawContours(img, contours, -1, Scalar(255,0,255), 2);
//    // -1 for all of the contours
//    /* void drawContours(InputOutputArray image, const vector<vector<Point>>& contours,
//    int contourIdx, const Scalar& color, int thickness = 1,
//    int lineType = LINE_8, InputArray hierarchy = noArray(),
//    int maxLevel = INT_MAX, Point offset = Point()); */
//
//    vector<vector<Point>> conPoly(contours.size());
//    vector<Rect> boundRect(contours.size());
//    string objectType;
//
//    for(int i = 0; i < contours.size(); i++) {
//        int area = contourArea(contours[i]);
//        cout << area << endl;
//        if(area > 1000) {
//            // bounding box
//            float peri = arcLength(contours[i], true);
//            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
//            /*contours[i]: This represents the i-th contour, which is a vector of points (typically obtained from the output of the findContours function).
//
//             conPoly[i]: This is an empty vector where the function will store the approximated polygonal representation of the i-th contour. After the function call, conPoly[i] will contain the points of the approximated polygon.
//
//             0.02 * peri: This is the epsilon parameter of the approximation. It determines the maximum distance between the original contour and the approximated polygon. A smaller epsilon value results in a more accurate approximation, while a larger epsilon value results in a simpler approximation with fewer points.
//
//             true: The last parameter specifies whether the approximated polygon should be a closed contour or not. When set to true, the function creates a closed polygon (connects the last point to the first), and when set to false, it creates an open polygon.
//
//             The approxPolyDP function approximates the contour represented by contours[i] using the Ramer-Douglas-Peucker algorithm. This algorithm simplifies a polygonal curve by removing intermediate points while preserving its shape within a specified tolerance (epsilon). It is commonly used to reduce the number of points in a contour while maintaining its overall shape, which can be helpful for further processing or analysis.
//
//             By using this function, you can obtain a simplified version of the contour stored in conPoly[i], which contains fewer points than the original contours[i] while still representing the essential shape of the object or region in the image.
//
//                */
//            cout << conPoly[i].size() << endl;
//            // makes a rectange around the shapes that have contours
//            boundRect[i] = boundingRect(conPoly[i]);
//            int objCor = (int)conPoly[i].size();
//            if(objCor == 3) {
//                objectType = "Triangle";
//            }
//            if(objCor == 4) {
//                float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
//                cout << aspRatio << endl;
//                if(aspRatio > 0.95 && aspRatio < 1.05) {
//                    objectType = "Square";
//                }
//                else {
//                    objectType = "Rectangle";
//                }
//            }
//            if(objCor > 4) {
//                objectType = "Circle";
//            }
//
//            drawContours(img, conPoly, i, Scalar(255,0,255), 2);
//            rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 5);
//            putText(img, objectType, {boundRect[i].x, boundRect[i].y - 5}, FONT_HERSHEY_PLAIN, 1, Scalar(0,69,255), 1.5);
//        }
//    }
//
//}
///*
// The findContours function is a powerful image processing function in OpenCV used to detect and extract contours from a binary image. A contour can be thought of as a curve joining all continuous points along the boundary of a connected component in the image, with the same color or intensity. These contours can represent various shapes, objects, or structures within the image.
//
// The syntax for the findContours function is as follows:
// void findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset = Point());
//
// Parameters:
//
// image: The input binary image where contours will be detected. This image should be a single-channel (grayscale) image where objects are represented by white pixels (255) and the background is represented by black pixels (0).
// contours: A vector of std::vector<cv::Point> that will store the detected contours. Each element of this vector corresponds to a single contour and contains a series of cv::Point points representing the contour's boundary.
// hierarchy: Optional output array or vector representing the hierarchy of contours. It provides information about nested contours and their relationships. You can set this parameter to cv::noArray() if you don't need it.
// mode: Specifies the contour retrieval mode. Common values are cv::RETR_EXTERNAL (retrieve only the external contours), cv::RETR_LIST (retrieve all the contours in a simple list), cv::RETR_TREE (retrieve all the contours in a hierarchical structure), etc.
// method: Specifies the contour approximation method. Common values are cv::CHAIN_APPROX_SIMPLE (compresses horizontal, vertical, and diagonal segments and leaves only their end points) and cv::CHAIN_APPROX_NONE (stores all the contour points).
// offset: Optional parameter that can be used to offset the detected contours' coordinates.
// After calling the findContours function, the detected contours will be stored in the contours vector, and you can use them for various purposes, such as drawing, shape analysis, or object recognition. The hierarchy array provides information about the relationship between contours, allowing you to navigate through nested contours and parent-child relationships.
// */

// Chapter 8 - Face Detection

//int main() {
//    string imagePath = "/Users/magen/Desktop/test.png";
//    Mat img = imread(imagePath);
//
//    CascadeClassifier faceCascade;
//    faceCascade.load("/Users/magen/Desktop/haarcascade_frontalface_default.xml");
//
//    if(faceCascade.empty()) {
//        cout << "XML file not loaded" << endl;
//    }
//
//    vector<Rect> faces;
//    faceCascade.detectMultiScale(img, faces, 1.1, 10);
//    // detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
//
//    for(int i = 0; i < faces.size(); i++) {
//        rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255,0,255), 3);
//    }
//
//    imshow("Image", img);
//    waitKey(0);
//
//    return 0;
//}

// Face Detection on Webcam
//
//int main() {
//    // Create a VideoCapture object to access the webcam (device 0)
//    VideoCapture cap(0);
//
//    // Check if the VideoCapture object is successfully opened
//    if (!cap.isOpened()) {
//        cout << "Error: Unable to access the webcam." << endl;
//        return -1;
//    }
//
//    // Replace "your_username" with your actual username
//    string imagePath = "/Users/magen/Desktop/test.png";
//    Mat img1 = imread(imagePath);
//
//    CascadeClassifier faceCascade;
//    faceCascade.load("/Users/magen/Desktop/haarcascade_frontalface_default.xml");
//
//    if(faceCascade.empty()) {
//        cout << "XML file not loaded" << endl;
//    }
//
//    // Create a Mat object to hold the webcam frame
//    Mat img2;
//
//    // Infinite loop to continuously read and display frames from the webcam
//    while (true) {
//        // Read a frame from the webcam and store it in the 'img' Mat object
//        cap.read(img2);
//
//        // Check if the frame is empty (indicating a problem with the webcam)
//        if (img2.empty()) {
//            cout << "Error: Webcam frame is empty." << endl;
//            break;
//        }
//
////        // Convert the frame to grayscale for face detection
////        Mat gray;
////        cvtColor(img2, gray, COLOR_BGR2GRAY);
//
//        // Perform face detection on the grayscale frame
//        vector<Rect> faces;
//        faceCascade.detectMultiScale(img2, faces, 1.1, 10);
//
//        // Draw rectangles around the detected faces
//        for (size_t i = 0; i < faces.size(); i++) {
//            rectangle(img2, faces[i].tl(), faces[i].br(), Scalar(0, 255, 0), 3);
//        }
//
//        // Display the current frame from the webcam in a window titled "Webcam"
//        imshow("Webcam", img2);
//
//        // Wait for 30 milliseconds (30ms) for a key press
//        int key = waitKey(30);
//
//        // Check if the user pressed 'q' (ASCII code 113)
//        if (key == 'q')
//            break;
//    }
//    // Close the window and release the webcam
//    destroyAllWindows();
//
//    return 0;
//}

/*
 
 for image box detection:
 converting the input image to grayscale is an essential step in the Viola-Jones face detection algorithm.

 The Viola-Jones algorithm is a machine learning-based approach for object detection, specifically designed for face detection. It uses Haar-like features to detect objects in images efficiently. The algorithm consists of two main stages: the Haar feature selection and the AdaBoost training.

 The Haar features are rectangular filters that are applied to the integral image of the input image. These features capture local intensity differences in various regions of the image. To compute these features efficiently, the input image needs to be in grayscale. Converting the image to grayscale reduces the computation required for feature extraction and comparison since grayscale images have only one channel.

 After extracting the Haar-like features from the integral image, the AdaBoost training stage selects the most relevant features and combines them to create a strong classifier for face detection. This classifier is then used to scan the input image at multiple scales and positions to detect faces.

 In summary, turning the image into grayscale is a critical preprocessing step in the Viola-Jones algorithm because it simplifies the feature extraction process and reduces the computational complexity of the algorithm. Grayscale images are easier to work with for Haar-like feature extraction, which is at the core of the Viola-Jones face detection method.*/

//// Project 1 - Virtual Painter
//Mat img;
//vector<vector<int>> newPoints; // {x, y(our center), color ie. 0 is purple and 1 is green}
//vector<vector<int>> findColor(Mat img);
//Point getContours(Mat imgDia);
//void drawOnCanvas(vector<vector<int>> newPoints, vector<Scalar> myColorValues);
//
//vector<vector<int>> myColors{
//    {0, 100, 100, 10, 255, 255}, //red
//    {100, 100, 0, 130, 255, 100}, // midnight blue
//    {35, 80, 100, 80, 255, 255} // green
//};
//
///* ie. purple
// H min: 124
// H max: 143
// S min: 48
// S max: 170
// V min: 117
// V max: 255
// */
//
//// 124,48,117,143,170,255 corresponds to purple
//// 68,72,156,102,126,255 corresponds to green
//// 0,62,0,35,255,255 corresponds to orange
//// 100,130,100,255,100,255 corresponds to blue
//vector<Scalar> myColorValues {{0,0,255}, // red
//    {255,0,0}, // blue
//    {0,255,0}}; // green
//
//int main() {
//    VideoCapture cap(0);
//    while (true) {
//        cap.read(img);
//        newPoints = findColor(img);
//        drawOnCanvas(newPoints, myColorValues);
//
//        imshow("Image", img);
//
//        int key = waitKey(1); // Wait for 1 millisecond
//        if (key == 'q') {
//            break;
//        }
//        else if (key == 'c') {
//            newPoints.clear(); // Clear the newPoints vector when 'c' is pressed
//            // clears out the vector
//        }
//    }
//
//    return 0;
//}
//
//// This function takes an input image 'img' and finds specific colors in it using HSV color space.
//
//vector<vector<int>> findColor(Mat img) {
//    Mat imgHSV;
//
//    // Convert the input image from BGR color space to HSV color space.
//    cvtColor(img, imgHSV, COLOR_BGR2HSV);
//
//    // Loop through the list of colors to be detected.
//    for(int i = 0; i < myColors.size(); i++){
//        // Extract lower and upper bounds of the current color from 'myColors' list.
//        Scalar lower(myColors[i][0], myColors[i][1], myColors[i][2]);
//        Scalar upper(myColors[i][3], myColors[i][4], myColors[i][5]);
//
//        // Create a mask using 'inRange' function to isolate the pixels of the current color.
//        Mat mask;
//        inRange(imgHSV, lower, upper, mask);
//
//        // Find the contours of the color regions in the mask and get a representative point.
//        Point myPoint = getContours(mask);
//
//        // Check if a valid point was detected (not the default origin point).
//        if(myPoint.x != 0 && myPoint.y != 0) {
//            // If a valid point was found, store its coordinates and the index of the detected color.
//            newPoints.push_back({myPoint.x, myPoint.y, i});
//        }
//    }
//
//    // Return a vector containing the detected points along with their associated color index.
//    return newPoints;
//}
//
//
//Point getContours(Mat imgDia)
//{
//    // each vector will be a contour and each contour will have some points
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//
//    findContours(imgDia, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//    // drawContours(img, contours, -1, Scalar(255,0,255), 2);
//    // -1 for all of the contours
//    /* void drawContours(InputOutputArray image, const vector<vector<Point>>& contours,
//    int contourIdx, const Scalar& color, int thickness = 1,
//    int lineType = LINE_8, InputArray hierarchy = noArray(),
//    int maxLevel = INT_MAX, Point offset = Point()); */
//
//    vector<vector<Point>> conPoly(contours.size());
//    vector<Rect> boundRect(contours.size());
//    Point myPoint(0,0);
//
//    for(int i = 0; i < contours.size(); i++) {
//        int area = contourArea(contours[i]);
//       // cout << area << endl;
//        if(area > 1000) {
//            // bounding box
//            float peri = arcLength(contours[i], true);
//            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
//            // makes a rectange around the shapes that have contours
//            boundRect[i] = boundingRect(conPoly[i]);
//            myPoint.x = boundRect[i].x + boundRect[i].width / 2;
//            // we want to draw from the center of the marker; NOT the edge
//            myPoint.y = boundRect[i].y;
//
//           // drawContours(img, conPoly, i, Scalar(255,0,255), 2);
//           // rectangle(img,boundRect[i].tl(),boundRect[i].br(),Scalar(0,255,0),5);
//        }
//    }
//    return myPoint;
//}
//
//void drawOnCanvas(vector<vector<int>> newPoints, vector<Scalar> myColorValues){
//    for(int i = 0; i < newPoints.size(); i++) {
//        circle(img, Point(newPoints[i][0], newPoints[i][1]), 10, myColorValues[newPoints[i][2]], FILLED);
//        // radius is 10
//    }
//}

//// Project 2 - Document Scanner
//Mat imgOriginal, imgGray, imgCanny, imgThre, imgBlur, imgDia, imgErode, imgWarp, imgCrop;
//vector<Point> initialPoints, docPoints;
//Mat preProcessing(Mat img);
//vector<Point> getContours(Mat imgDia);
//void drawPoints(vector<Point> points, Scalar color);
//vector<Point> reorder(vector<Point> points);
//Mat getWarp(Mat img, vector<Point> points, float w, float h);
//
//float w = 420, h = 596;
//
//int main() {
//    string path = "/Users/magen/Desktop/free-letter-mockup.jpg";
//    imgOriginal = imread(path); // read an image file and load it as a matrix into memory
//    // resize(imgOriginal, imgOriginal, Size(), 0.75, 0.75);
//    // void cv::resize(InputArray src, OutputArray dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);
//    // we scale it down
//
//    // 3 steps we will follow:
//    // Preprocessing
//    imgThre = preProcessing(imgOriginal);
//
//    // Get Contours - Biggest
//    initialPoints = getContours(imgThre);
//    // drawPoints(initialPoints, Scalar(0,0,255));
//    docPoints = reorder(initialPoints);
//    // drawPoints(docPoints, Scalar(0,255,0));
//
//    // Warp
//    imgWarp = getWarp(imgOriginal, docPoints, w, h);
//
//    // Crop
//    int cropVal = 10;
//    Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
//    imgCrop = imgWarp(roi); // extracts submatrix
//    // roi means region of interest
//    // takes in x, y, width, height
//
//    imshow("Image", imgOriginal);
//    // imshow("ImgDia", imgThre);
//    imshow("ImgWarp", imgWarp);
//    imshow("ImgCrop", imgCrop);
//    waitKey(0);
//
//    return 0;
//}
//
//Mat preProcessing(Mat img) {
//    cvtColor(img, imgGray, COLOR_BGR2GRAY);
//    //  GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY)
//    GaussianBlur(imgGray, imgBlur, Size(7,7), 5,0);
//    Canny(imgBlur, imgCanny, 50, 150);
//    // 50 and 150 are threshold values
//    // low threshold - weak edges
//    // high threshold - strong edges
//    // black areas are blow the low and high thresholds
//
//    Mat kernal = getStructuringElement(MORPH_RECT, Size(5,5));
//    // lower decreases dilation, higher increases
//
//    dilate(imgCanny, imgDia, kernal); // increases thickness
//    // in python we use NUM_PI to define our kernal, since 3rd argument takes a kernal
//    // erode(imgDia, imgErode, kernal); // decreases thickness
//
//    return imgDia;
//}
//
//vector<Point> getContours(Mat imgDia)
//{
//    // each vector will be a contour and each contour will have some points
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//
//    findContours(imgDia, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//    // drawContours(img, contours, -1, Scalar(255,0,255), 2);
//    // -1 for all of the contours
//    /* void drawContours(InputOutputArray image, const vector<vector<Point>>& contours,
//    int contourIdx, const Scalar& color, int thickness = 1,
//    int lineType = LINE_8, InputArray hierarchy = noArray(),
//    int maxLevel = INT_MAX, Point offset = Point()); */
//
//    vector<vector<Point>> conPoly(contours.size());
//    vector<Rect> boundRect(contours.size());
//    vector<Point> biggest;
//    int maxArea;
//
//    for(int i = 0; i < contours.size(); i++) {
//        int area = contourArea(contours[i]);
//       // cout << area << endl;
//        if(area > 1000) {
//            // bounding box
//            float peri = arcLength(contours[i], true);
//            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
//            // makes a rectange around the shapes that have contours
//
//            if(area > maxArea && conPoly[i].size() == 4) {
//               // drawContours(imgOriginal, conPoly, i, Scalar(255,0,255), 5);
//                biggest = {conPoly[i][0],conPoly[i][1],conPoly[i][2],conPoly[i][3]};
//                maxArea = area;
//            }
//
//           // drawContours(imgOriginal, conPoly, i, Scalar(255,0,255), 2);
//          // rectangle(imgOriginal,boundRect[i].tl(),boundRect[i].br(),Scalar(0,255,0),5);
//        }
//    }
//    return biggest;
//}
//
//void drawPoints(vector<Point> points, Scalar color) {
//    for(int i = 0; i < points.size(); i++) {
//        circle(imgOriginal, points[i], 10, color, FILLED);
//        // size of circle is 30
//        putText(imgOriginal, to_string(i), points[i], FONT_HERSHEY_PLAIN, 4, color, 4);
//        // scale color thickness
//    }
//
//}
//
//vector<Point> reorder(vector<Point> points) {
//    vector<Point> newPoints;
//    vector<int> sumPoints, subPoints;
//    for(int i = 0; i < 4; i++) {
//        sumPoints.push_back(points[i].x + points[i].y);
//        subPoints.push_back(points[i].x - points[i].y);
//    }
//        newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // top-left corner
//        newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // top-right corner.
//        newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // bottom-left corner.
//        newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // bottom-right corner
//
//        /*
//        void cv::minMaxLoc(
//        cv::InputArray src,        // Input array (single-channel)
//        double* minVal,            // Pointer to store the minimum value
//        double* maxVal = 0,        // Pointer to store the maximum value
//        cv::Point* minLoc = 0,     // Pointer to store the location of the minimum value
//        cv::Point* maxLoc = 0,     // Pointer to store the location of the maximum value
//        cv::InputArray mask = noArray() // Optional mask used to restrict the search area
//         );
//
//         */
//    return newPoints;
//}
//
//Mat getWarp(Mat img, vector<Point> points, float w, float h) {
//    Point2f src[4] = {points[0], points[1], points[2], points[3]};
//    Point2f dst[4] = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
//
//    /*
//     {0.0f, 0.0f}: The top-left corner of the transformed image.
//     {w, 0.0f}: The top-right corner of the transformed image.
//     {0.0f, h}: The bottom-left corner of the transformed image.
//     {w, h}: The bottom-right corner of the transformed image.
//     */
//
//    Mat matrix = getPerspectiveTransform(src, dst);
//    warpPerspective(img, imgWarp, matrix, Point(w,h));
//
//    return imgWarp;
//}
//
///*
// MORPH_RECT (Rectangle Structuring Element):
//
// MORPH_RECT represents a rectangular structuring element. It's often used to perform dilation and erosion operations. When you apply dilation with a rectangle-shaped structuring element, it expands the bright regions in the image. When you apply erosion, it shrinks the bright regions.
//
// Visually, it means dilation with a rectangle structuring element makes bright regions larger and more connected, while erosion makes them smaller and less connected.
//
// MORPH_CROSS (Cross Structuring Element):
//
// MORPH_CROSS represents a cross-shaped structuring element. Like the rectangle, it's used for dilation and erosion, but it can yield slightly different results. When applying dilation, a cross-shaped structuring element expands the bright regions, similar to the rectangle. However, during erosion, the cross may have a slightly different effect compared to the rectangle due to its shape.
//
// The cross structuring element is often used when you want to emphasize diagonal connections or when the diagonal connectivity of objects matters more than the horizontal and vertical connectivity.
// */

