#include "video.h"
#include "Opencv.h"
#include "Openmp.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <omp.h>
#include<ipp.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void video::MultipleVideoProcessingTest()
{
#pragma omp parallel sections
    {
#pragma omp section
        DisplayVideo("person.mp4", "original");
#pragma omp section
        DisplayVideo("person.mp4", "gaussian");
#pragma omp section
        DisplayVideo("person.mp4", "sobel");
#pragma omp section
        DisplayVideo("person.mp4", "gabor");
    }
}

int video::DisplayVideo(string strVideo, string windowName)
{
    VideoCapture cap(strVideo);
    if (!cap.isOpened()) return -1;

    namedWindow(windowName, 1);

    double fstart, fend, fprocTime;
    double fps;
    int f = 0;
    for (;;)
    {
        f++;
        fstart = omp_get_wtime();

        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }

        if (windowName == "original")
            ;
        else if (windowName == "gaussian")
            GaussianBlur(frame, frame, Size(5, 5), 10, 10);
        else if (windowName == "sobel")
            Sobel(frame, frame, -1, 0, 1);
        else if (windowName == "gabor")
        {
            Mat kernel = getGaborKernel(Size(21, 21), 5, 1, 10, 1, 0, CV_32F);
            filter2D(frame, frame, -1, kernel);
        }

        fend = omp_get_wtime();
        fprocTime = fend - fstart;
        fps = 1 / fprocTime;
        putText(frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 3);
        imshow(windowName, frame);
        waitKey(10);
    }
    cout << f << endl;
    return 0;
}

void video::Detection(string strVideo)
{
    VideoCapture cap(strVideo);
    if (!cap.isOpened()) return;
  
    double fstart, fend, fprocTime;
    double fps;
  
    vector<Mat> frames;

    for (;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frames.push_back(frame);
    }

#pragma omp parallel for
    for (int i = 0; i < frames.size(); i++)
    {
        Mat frame = frames[i];

        if (frame.empty()) break;
        
        if (i % 3 == 0)
        {
            Display_Original(frame, "0");
        }
        if (i % 3 == 1)
        {
            Display_Face(frame, "1");
        }
        if (i % 3 == 2)
        {
            Display_Eye(frame, "2");
        }
    }

}


void video::Display_Original(Mat cap_frame, string windowName)
{
    namedWindow(windowName, 1);

    double fstart, fend, fprocTime;
    double fps;
    fstart = omp_get_wtime();
    fend = omp_get_wtime();
    fprocTime = fend - fstart;
    fps = 1 / fprocTime;
    putText(cap_frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 3);
    imshow(windowName, cap_frame);
    waitKey(20);
}

void video::Display_Face(Mat cap_frame, string windowName)
{
    namedWindow(windowName, 1);

    FaceDetect(cap_frame);
    imshow(windowName, cap_frame);
    waitKey(20);   
}

void video::Display_Eye(Mat cap_frame, string windowName)
{
    namedWindow(windowName, 1);

    EyeDetect(cap_frame);
    imshow(windowName, cap_frame);
    waitKey(20);
   
}

void video::FaceDetect(Mat cap_frame)
{
    CascadeClassifier face_cascade;
    string cascadepath_face = "D:\\soojie\\program\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";

    if (!face_cascade.load(cascadepath_face))
    {
        cout << "cascade load error\n";
        return;
    }

    double fstart, fend, fprocTime;
    double fps;

    fstart = omp_get_wtime();

    vector<Rect> faces;
    
    face_cascade.detectMultiScale(cap_frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        rectangle(cap_frame, Rect(center.x - (faces[i].width / 2), center.y - (faces[i].height / 2), faces[i].width, faces[i].height), Scalar(0, 0, 255), 3, 8, 0);
    }

    fend = omp_get_wtime();
    fprocTime = fend - fstart;
    fps = 1 / fprocTime;
    putText(cap_frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 3);

}

void video::EyeDetect(Mat cap_frame)
{
    CascadeClassifier eye_cascade;
    string cascadepath_eye = "D:\\soojie\\program\\opencv\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml";

    if (!eye_cascade.load(cascadepath_eye))
    {
        cout << "cascade load error\n";
        return;
    }

    double fstart, fend, fprocTime;
    double fps;

    fstart = omp_get_wtime();

    vector<Rect> eyes;

    eye_cascade.detectMultiScale(cap_frame, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

    for (size_t i = 0; i < eyes.size(); i++)
    {
        Point center(eyes[i].x + eyes[i].width / 2, eyes[i].y + eyes[i].height / 2);
        rectangle(cap_frame, Rect(center.x - (eyes[i].width / 2), center.y - (eyes[i].height / 2), eyes[i].width, eyes[i].height), Scalar(255, 0, 0), 3, 8, 0);
    }

    fend = omp_get_wtime();
    fprocTime = fend - fstart;
    fps = 1 / fprocTime;
    putText(cap_frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 3);

}
