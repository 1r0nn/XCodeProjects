# XCodeProjects 

I first worked on knowing and studying how each function works in OpenCV to become familiar on how to build functional applications. I then worked on the relevant projects listed below. These are all done using the OpenCV library via C++ language.

## (a) Virtual Painter
- Utilizes colors recognized from a camera feed for virtual drawing.
- HSV color space employed to isolate specific colors.
- Binary masks generated through contour extraction to define colored regions.
- Central points of each color area analyzed as contours merge.
- Combines color recognition, contour detection, and interactive drawing.
- Results in an engaging coloring experience for users.

## (b) Facial Detection 
- Utilizes the faceCascade technique for face detection in static images and real-time webcam frames.
- Loads the pre-trained Haar Cascade classifier designed for frontal face detection.
- Implements the detectMultiScale function to draw rectangles around identified faces within the camera frame.
- 
## (c) Document Scanner
- System retrieves planar documents from different angles or orientations.
- Process includes:
  - Initial image preprocessing.
  - Contour recognition of the document.
  - Perspective transformation of the image.
  - Precise cropping for an improved document image.
- Scanned document is returned in its correct perspective.
- Provides valuable functionality for applications requiring these capabilities.
