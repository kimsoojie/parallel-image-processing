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
        DisplayVideo("testvideo.mp4", "original");
#pragma omp section
        DisplayVideo("testvideo.mp4", "gaussian");
#pragma omp section
        DisplayVideo("testvideo.mp4", "sobel");
#pragma omp section
        DisplayVideo("testvideo.mp4", "gabor");
    }
}

int video::DisplayVideo(string strVideo, string windowName)
{
    VideoCapture cap(0);
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

void video::FaceDetectTest()
{
    //FaceDetect("person.mp4", "original");
    BodyDetect("person.mp4", "original");

}

void video::FaceDetect(string strVideo, string windowName)
{
    VideoCapture cap(strVideo);
    if (!cap.isOpened());

    namedWindow(windowName, 1);

    double fstart, fend, fprocTime;
    double fps;

    CascadeClassifier face_cascade;
    //string cascadepath_face = "D:\\program\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";
    string cascadepath_face = "D:\\soojie\\program\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";

    if (!face_cascade.load(cascadepath_face))
    {
        cout << "cascade load error\n";
        return;
    }

    for (;;)
    {
        fstart = omp_get_wtime();

        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }

        vector<Rect> faces;
        Mat frame_gray;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (size_t i = 0; i < faces.size(); i++)
        {
            Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            rectangle(frame, Rect(center.x-(faces[i].width / 2), center.y-(faces[i].height / 2), faces[i].width, faces[i].height), Scalar(0, 0, 255), 4, 8, 0);
        }

        fend = omp_get_wtime();
        fprocTime = fend - fstart;
        fps = 1 / fprocTime;
        putText(frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 3);
        imshow(windowName, frame);
        waitKey(10);
    }
}

void video::BodyDetect(string strVideo, string windowName)
{
    VideoCapture cap(strVideo);
    if (!cap.isOpened());

    namedWindow(windowName, 1);

    double fstart, fend, fprocTime;
    double fps;

    CascadeClassifier eye_cascade;
    //string cascadepath_body = "D:\\program\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_fullbody.xml";
    string cascadepath_eye = "D:\\soojie\\program\\opencv\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml";

    if (!eye_cascade.load(cascadepath_eye))
    {
        cout << "cascade load error\n";
        return;
    }

    for (;;)
    {
        fstart = omp_get_wtime();

        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }

        vector<Rect> eyes;
        Mat frame_gray;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        eye_cascade.detectMultiScale(frame_gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (size_t i = 0; i < eyes.size(); i++)
        {
            Point center(eyes[i].x + eyes[i].width / 2, eyes[i].y + eyes[i].height / 2);
            rectangle(frame, Rect(center.x - (eyes[i].width / 2), center.y - (eyes[i].height / 2), eyes[i].width, eyes[i].height), Scalar(0, 0, 255), 4, 8, 0);
        }

        fend = omp_get_wtime();
        fprocTime = fend - fstart;
        fps = 1 / fprocTime;
        putText(frame, "fps: " + to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 3);
        imshow(windowName, frame);
        waitKey(10);
    }
}
