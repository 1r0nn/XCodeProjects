# XCodeProjects 

I first worked on knowing and studying how each function works in OpenCV to become familiar on how to build functional applications. I then worked on the relevant projects listed below. These are all done using the OpenCV library via C++ language.

## (a) Virtual Painter
By harnessing distinct colors recognized from a camera feed, users are enabled to virtually draw. The HSV color space is utilized to isolate specific colors, generating binary masks through contour extraction that defines regions of color. The central points of each color area are subsequently analyzed as contours are merged, effectively combining color recognition, contour detection, and interactive drawing. This results in a captivating coloring experience that engages users.

## (b) Facial Detection 
Utilizing the faceCascade technique, faces are detected within static images and real-time webcam frames. The pre-trained Haar Cascade classifier, specialized in frontal face detection, is loaded for this purpose. Employing the detectMultiScale function, rectangles are drawn around each face identified within the camera frame.

## (c) Document Scanner
Through a series of steps, the system retrieves planar documents captured from various angles or orientations. The process involves initial image preprocessing, contour recognition of the document, perspective transformation of the image, and finally, precise cropping to yield an enhanced document image. Following the scanning, the system returns the scanned document in its accurate perspective. This project delivers valuable functionality for applications with a need for such capabilities.
