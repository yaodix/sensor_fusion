#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType,
                      std::vector<double>& time)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type() != CV_32F)
        {
            // OpenCV bug workaround: convert binary descriptors to floating point
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        time.push_back(1000 * t / 1.0);
        cout << "(NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;

        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time.push_back(1000 * t / 1.0);
        cout << "(KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        double minDescDistRatio = 0.8;
        for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if( (*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, std::vector<double>& time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time.push_back(1000 * t / 1.0);
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::vector<double>& time, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time.push_back(1000 * t / 1.0);
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Harris Corner Detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::vector<double>& time, bool bVis)
{
    // Harris detector params
    int blockSize = 2;      // for everypixel, a blockSize Ã - blockSize neighborhood is considered
    int apertureSize = 3;   // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100;  // minimum value for a corner in the 8-bit scaled response matrix
    double k = 0.04;        // Harris parameter

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maximum suppression
    for(size_t j = 0; j < dst_norm.rows; j++)
    {
        for(size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if(response > minResponse)
            {
                // store points above a threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for(auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if(kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if(newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                if(!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time.push_back(1000 * t / 1.0);
    cout << "Harris Corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

}

// Modern Algorithms Keypoint Detectors
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, std::vector<double>& time, bool bVis)
{
    // Select keypoint detector
    if(detectorType.compare("FAST")== 0)
    {
        // Define FAST params
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

        // Detect keyponts
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time.push_back(1000 * t / 1.0);
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    if(detectorType.compare("BRISK")== 0)
    {
        // Define BRISK detector
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

        // Detect keypoints
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time.push_back(1000 * t / 1.0);
        cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    if(detectorType.compare("ORB")== 0)
    {
        // Define ORB detector
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

        // Detect keypoints
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time.push_back(1000 * t / 1.0);
        cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    if(detectorType.compare("AKAZE")== 0)
    {
        // Define AKAZE detector
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();

        // Detect keypoints
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);  
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time.push_back(1000 * t / 1.0);
        cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;      
    }
    if(detectorType.compare("SIFT")== 0)
    {
         // Define SIFT detector
        cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();

        // Detect keypoints
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);  
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time.push_back(1000 * t / 1.0);
        cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;    
    }
}