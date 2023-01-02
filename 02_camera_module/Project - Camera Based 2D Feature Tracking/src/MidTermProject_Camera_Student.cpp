/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    ofstream DataLog("Midterm_Data_Log.csv");
    vector<string> titles = {"Image ID", "Detector", "Descriptor", "Total Keypoints", "Vehicle Keypoints", "Detector Runtime", "Descriptor Runtime", "Matched Keypoints", "Matcher Runtime"};
    for(int i = 0; i < titles.size(); i++)
    {
        DataLog << titles[i];
        if(i != (titles.size() - 1)) DataLog << ", ";
    }
    DataLog << "\n";

    vector<string> detectorTypes = {"HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    for(auto detType : detectorTypes)
    {// DETECTOR LOOP

        for(auto descType : descriptorTypes)
        {// DESCRIPTOR LOOP
            
            /* MAIN LOOP OVER ALL IMAGES */
            dataBuffer.clear();
            DataLog << "\n";

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                if(dataBuffer.size() == dataBufferSize)
                {
                    dataBuffer[0] = dataBuffer[1]; //cycle the ring buffer
                    dataBuffer[1] = frame;
                }
                if(dataBuffer.size() < dataBufferSize)
                {
                    dataBuffer.push_back(frame); //initally populate empty vector with 2 pushbacks
                }

                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                // Initialize data vector for updating CSV
                vector<string> dataLogger;
                vector<double> time;

                dataLogger.push_back(to_string(imgIndex)); // update Image ID

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                string detectorType = detType;
                dataLogger.push_back(detectorType);
                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, time, false);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, time, false);
                }
                else
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, time, false);
                }
                dataLogger.push_back(to_string(keypoints.size())); //Total Number of Keypoints
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                if (bFocusOnVehicle)
                {
                    // Define Crop Region
                    cv::Rect vehicleRect(535, 180, 180, 150);
                    vector<cv::KeyPoint> cropPoints;

                    // Filter points
                    for(auto point : keypoints)
                    {
                        if(vehicleRect.contains(cv::Point2f(point.pt))) { cropPoints.push_back(point); }
                    }

                    // Assign cropped KeyPoints
                    keypoints = cropPoints;
                    dataLogger.push_back(to_string(keypoints.size())); // Vehicle Keypoints
                    cout << "The number of keypoints in Image " << imgIndex << " is " << keypoints.size() << endl;
                }

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;

                string descriptorType = descType; // Select: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
                
                // AKAZE cant be paired with other algorithms, only KAZE/AKAZE keypoints work with KAZE/AKAZE algorithms
                if(descType.compare("AKAZE") == 0 && detType.compare("AKAZE") != 0) {break;}
                // SIFT keypoints also do not process well with other descriptors: experimentally SIFT-ORB do not work due to octave values
                if(detType.compare("SIFT") == 0 && descType.compare("ORB") == 0) {break;}

                dataLogger.push_back(descriptorType);
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, time);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                // Initial Row update for image 0
                if (dataBuffer.size() == 1)
                {
                    if(dataLogger.size() != 6) {dataLogger.push_back("0");}
                    swap(dataLogger.at(2), dataLogger.at(4));
                    swap(dataLogger.at(3), dataLogger.at(4));
                    
                    vector<string> timeS;
                    for(int i = 0; i < time.size(); i++) {timeS.push_back(to_string(time[i]));}
                    if(time.size() != 3) {timeS.push_back("0");}
                    auto iter = dataLogger.insert(dataLogger.begin() + 5, timeS.begin(), timeS.end());

                    for(int i = 0; i < dataLogger.size(); i++)
                    {
                        DataLog << dataLogger[i];
                        if(i != (dataLogger.size() - 1)) DataLog << ", ";
                    }
                    DataLog << "\n";
                    dataLogger.clear();
                    time.clear();
                    timeS.clear();
                }

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */
                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    string descriptorType = descType.compare("SIFT") == 0 ? "DES_HOG" : "DES_BINARY"; // DES_BINARY, DES_HOG
                    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType, time);
                    dataLogger.push_back(to_string(matches.size()));

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(3000); // wait for key to be pressed
                    }
                    bVis = false;

                    // Organize dataLogger and time into CSV format with given Titles
                    swap(dataLogger.at(2), dataLogger.at(4));
                    swap(dataLogger.at(3), dataLogger.at(4));
                    vector<string> timeS;
                    for(int i = 0; i < time.size(); i++) {timeS.push_back(to_string(time[i]));}
                    auto it = dataLogger.insert(dataLogger.begin() + 5, timeS.begin(), timeS.end());
                    
                    swap(dataLogger.at(7), dataLogger.at(8));

                    for(int i = 0; i < dataLogger.size(); i++)
                    {
                        DataLog << dataLogger[i];
                        if(i != (dataLogger.size() - 1)) DataLog << ", ";
                    }
                    DataLog << "\n";
                    dataLogger.clear();
                    time.clear();
                    timeS.clear();
                }

            } // eof loop over all images

        }// eof over all descriptor types

    }// eof over all detector types
    DataLog.close();

    return 0;
}
