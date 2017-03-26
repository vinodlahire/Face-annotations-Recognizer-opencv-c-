

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face/facerec.hpp"
#include "opencv2/face.hpp"
#include "opencv2/face/predict_collector.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;



int main(int argc, const char *argv[]) {
    // Get the path to your CSV:
    string fn_haar = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
    // holds images and labels
vector<Mat> images;
vector<int> labels;
string out;
    string stringStream;

// images for first person
images.push_back(imread("C:/person0/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
images.push_back(imread("C:/person0/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
images.push_back(imread("C:/person0/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
images.push_back(imread("C:/person0/3.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
images.push_back(imread("C:/person0/4.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
// images for second person
images.push_back(imread("C:/person1/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
images.push_back(imread("C:/person1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
images.push_back(imread("C:/person1/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
images.push_back(imread("C:/person1/3.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
images.push_back(imread("C:/person1/4.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<cv::face::FaceRecognizer> model = cv::face::createFisherFaceRecognizer();
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(0);
    // Holds the current frame from the Video device:
    Mat frame;
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:

            //std::string copyOfStr = stringStream.str();
            if (prediction==1)
            //out="Veeru";
            stringStream = "Veeru";
            else
            //out="Vinod";
            stringStream = "Vinod";
            string box_text = format("Prediction = %s", stringStream.c_str());
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}
