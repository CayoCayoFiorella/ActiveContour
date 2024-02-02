#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include "../headers/gvfc.h"
#include "../headers/common.h"
#include "../headers/drlse_edge.h"

using namespace cv;
using namespace std;

Mat img0;       // color image
Mat img0_c;     // copied color image
Mat res;        // result

Mat img1;       // gray-scale image
Mat mask;
Mat mask2; // Reemplace CvMat con cv::Mat

Point prev_pt = {-1, -1};

int Thresholdness = 141;

void on_mouse(int event, int x, int y, int flags, void* param)
{
    if (img0.empty())
        return;

    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prev_pt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prev_pt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt = Point(x, y);
        if (prev_pt.x < 0)
            prev_pt = pt;
        line(mask, prev_pt, pt, Scalar::all(255), 2, LINE_8, 0);
        line(img0, prev_pt, pt, Scalar(0, 0, 255), 2, LINE_8, 0);
        prev_pt = pt;
        imshow("Input Image", img0);
    }
}

void bind_line()
{
    cout << "     *------------------------------------------------------------------*" << endl;
}

void info_key0(int option)
{
    bind_line();
    cout << "     | Hot keys:                                                        |" << endl;
    cout << "     | Press 'ESC' - exit the program                                   |" << endl;
    switch (option)
    {
    case 1:
        cout << "     | Press 'w' or 'ENTER' - run GVFsnake (with loaded initial-contour)|" << endl;
        break;
    default:
        cout << "     | Press 'w' or 'ENTER' - run DRLSE (with loaded initial-contour)  |" << endl;
        break;
    }
    bind_line();
}

void info_key1(int option)
{
    bind_line();
    cout << "     | Hot keys:                                                        |" << endl;
    cout << "     | Press 'ESC' - exit the program                                   |" << endl;
    cout << "     | No mask is found                                                 |" << endl;
    switch (option)
    {
    case 1:
        cout << "     | Press 'w' or 'ENTER' - run GVFsnake (with default initial-contour)|" << endl;
        break;
    default:
        cout << "     | Press 'w' or 'ENTER' - run DRLSE (initial-contour required)     |" << endl;
        break;
    }
    bind_line();
}

void info_key2(int option)
{
    switch (option)
    {
    case 1:
        cout << "     | (Otherwise, before running, roughly mark the areas on the image) |" << endl;
        break;
    default:
        cout << "     | (Before running, roughly mark the areas on the image)            |" << endl;
        break;
    }
    cout << "     | Press 'r' - resort the original image                            |" << endl;
    bind_line();
}

int main(int argc, char* argv[])
{
    Size size; // Reemplace CvSize con cv::Size
    Point* point = nullptr; // Reemplace CvPoint con cv::Point y use nullptr

    Mat storage; // Reemplace CvMemStorage con cv::Mat o cv::Mat_<float>
    Mat contours; // Para almacenar las coordenadas de los contornos
    int length = 14, alg_option = 1, timestep = 5;
    float alpha = 0.05f, beta = 0.1f, gamma = 1.0f, kappa = 2.0f, flag = 0.0f, t;
    double lambda = 5.0f, epsilon = 1.5f, alfa = 1.5f;
    bool IS_MASK = false;

    if (argc < 3)
    {
        cout << "WARNING: Please locate the input image" << endl;
        return 0;
    }
    if (strcmp(argv[1], "-1") == 0)
        alg_option = 1;
    else if (strcmp(argv[1], "-2") == 0)
        alg_option = 2;
    else if (strcmp(argv[1], "-3") == 0)
        alg_option = 3;
    else
    {
        cout << "WARNING: Please choose one of the following two active contour methods: " << endl;
        cout << "-1: Gradient Vector Field Snake" << endl;
        cout << "-2: Distance Regularized Level Set Evolution (Expand curve)" << endl;
        cout << "-3: Distance Regularized Level Set Evolution (Shrink curve)" << endl;
    }

    img0 = imread(argv[2], IMREAD_COLOR);
    if (img0.empty())
    {
        cout << "ERROR: No Input Image Found" << endl;
        return 0;
    }

    size = img0.size();
    img0_c = Mat(size, CV_8UC3);
    res = Mat(size, CV_8UC3);
    img1 = Mat(size, CV_8UC1);
    mask = Mat(size, CV_8UC1);
    mask2 = Mat(size, CV_32FC1);

    cvtColor(img0, img1, COLOR_BGR2GRAY);
    img0.copyTo(img0_c);
    img0.copyTo(res);
    mask.setTo(0);

    namedWindow("Input Image", WINDOW_NORMAL);
    moveWindow("Input Image", 0, 0);

    if (argc == 4)
    {
        mask = imread(argv[3], IMREAD_GRAYSCALE);
        if (!mask.data)
        {
            system("clear");
            info_key0(alg_option);
        }
    }
    else
    {
        system("clear");
        info_key1(alg_option);
        info_key2(alg_option);
    }

    setMouseCallback("Input Image", on_mouse, 0);

    while (true)
    {
        char c = waitKey(0);

        if (c == 27)
            break;

        if (c == 'r')
        {
            mask.setTo(0);
            img0_c.copyTo(img0);
            img0_c.copyTo(res);
            imshow("Input Image", img0);
        }

        if (c == 'w' || c == '\n')
        {
            if (countNonZero(mask) > 0)
            {
                Ptr<Mat> storage = makePtr<Mat>();

                vector<vector<Point>> contours;
                vector<Point> contour;
                findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                if (contours.empty())
                    return 0;
                contour = contours[0];
                length = static_cast<int>(contour.size());
                if (length < 10)
                    return 0;
                point = new Point[length];
                for (int i = 0; i < length; i++)
                {
                    point[i] = contour[i];
                }

                if (IS_MASK)
                {
                    for (int i = 0; i < length; i++)
                    {
                        int j = (i + 1) % length;
                        line(img0, point[i], point[j], Scalar(255, 0, 0), 2, LINE_8, 0);
                    }
                    imshow("Input Image", img0);
                }
            }
            else
            {
                if (alg_option == 1)
                {
                    float t = 0.0f;
                    point = new Point[length];
                    for (int i = 0; i < length; i++)
                    {
                        point[i].x = int(float(size.width >> 1) +
                                         float(MIN(size.width, size.height) >> 2) * sinf(t));
                        point[i].y = int(float(size.height >> 1) +
                                         float(MIN(size.width, size.height) >> 2) * cosf(t));
                        if (i == length - 1)
                        {
                            point[i].x = point[0].x;
                            point[i].y = point[0].y;
                        }
                        t += 0.5f;
                    }
                    for (int i = 0; i < length; i++)
                    {
                        int j = (i + 1) % length;
                        line(img0, point[i], point[j], Scalar(255, 0, 0), 2, LINE_8, 0);
                    }
                    imshow("Input Image", img0);
                }
                else
                {
                    cout << "WARNING: before running, roughly mark the areas on the image" << endl;
                    continue;
                }
            }

            waitKey(0); // Espera hasta que se presione una tecla antes de continuar

            t = (float)getTickCount() - t;
            if (alg_option == 1)
                point = cvSnakeImageGVF(&img1, point, &length, alpha, beta, gamma, kappa, 50, 10, CV_REINITIAL, CV_GVF);

            else if (alg_option == 2)
                point = cvDRLSE(img1, mask, &length, lambda, alfa, epsilon, timestep, 200, 5, CV_LSE_EXP);

            else
                point = cvDRLSE(img1, mask, &length, lambda, alfa, epsilon, timestep, 200, 5, CV_LSE_SHR);

            t = (float)getTickCount() - t;
            if (!point)
            {
                cout << "Warning: Make sure the initial contour is closed" << endl;
                cout << "Press 'r' to resort the original image, then try again" << endl;
                continue;
            }

            cout << "exec time = " << t / (getTickFrequency() * 1e6) << endl;

            for (int i = 0; i < length; i++)
            {
                int j = (i + 1) % length;
                line(res, point[i], point[j], Scalar(0, 0, 255), 2, LINE_8, 0);
            }

            imshow("Result", res);

            string str = argv[2];
            string _str0 = "_ini";
            string _str1 = "_res";
            string _ext = ".png";

            _str0.insert(0, str, 0, str.length() - 4);
            _str0.insert(_str0.length(), _ext);

            _str1.insert(0, str, 0, str.length() - 4);
            _str1.insert(_str1.length(), _ext);

            imwrite(_str0.c_str(), img0);
            imwrite(_str1.c_str(), res);

            delete[] point;
        }
    }

    destroyAllWindows();
    return 0;
}

