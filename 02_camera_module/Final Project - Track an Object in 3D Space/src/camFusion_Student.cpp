
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, string windowName)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-0, bottom+50), cv::FONT_ITALIC, 0.5, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-0, bottom+125), cv::FONT_ITALIC, 0.5, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    // string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(1); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, 
        std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<double> distance;

    /*  kptMatches includes mismatches in the data as seen in previous visualizations; therefore,
        a method for determining a mismatch is to look at the average displacement of all keypoints
        as they spacially shift from one frame to the next.

        Taking the average distance will account for correct matches and remove mismatched points as
        the norm of the vectors (L2-Norm) will be much larger than the average
    */

    for(auto point : kptMatches)
    {
        const auto &kptCurr = kptsCurr.at(point.trainIdx);
        if(!boundingBox.roi.contains(kptCurr.pt)) {continue;}   // evaluate before storing next keypoint
        const auto &kptPrev = kptsPrev.at(point.queryIdx);

        distance.push_back(cv::norm(kptCurr.pt - kptPrev.pt));
    }

    int distanceSize = distance.size();
    double distanceMean = std::accumulate(distance.begin(), distance.end(), 0.0) / distanceSize;
    double scaledDistanceMean = 1.2 * distanceMean;

    for(auto point : kptMatches)
    {
        const auto &kptCurr = kptsCurr.at(point.trainIdx);

        if(!boundingBox.roi.contains(kptCurr.pt)) {continue;}

        const auto &kptPrev = kptsPrev.at(point.queryIdx);

        if(cv::norm(kptCurr.pt - kptPrev.pt) < scaledDistanceMean)
        {
            boundingBox.keypoints.push_back(kptCurr);
            boundingBox.kptMatches.push_back(point);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    // double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // compute camera-based TTC from median of distance ratios
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    if(isinf(-dT / (1 - medDistRatio)))
        TTC = NAN;
    else{
        TTC = -dT / (1 - medDistRatio);
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1/frameRate;
    double medianXPrev = 0.0;
    double medianXCurr = 0.0;
    std::vector<double> sortXPrev;
    std::vector<double> sortXCurr;

    for(auto point : lidarPointsPrev) {sortXPrev.push_back(point.x);}
    for(auto point : lidarPointsCurr) {sortXCurr.push_back(point.x);}

    sort(sortXPrev.begin(), sortXPrev.end());
    sort(sortXCurr.begin(), sortXCurr.end());

    medianXPrev = (sortXPrev.size() % 2 == 0) ? (sortXPrev[sortXPrev.size()/2 - 1] + sortXPrev[sortXPrev.size()/2]) / 2 : sortXPrev[sortXPrev.size()/2];
    medianXCurr = (sortXCurr.size() % 2 == 0) ? (sortXCurr[sortXCurr.size()/2 - 1] + sortXCurr[sortXCurr.size()/2]) / 2 : sortXCurr[sortXCurr.size()/2];
    
    if(fabs(medianXPrev - medianXCurr) <= max(medianXPrev,medianXCurr)*__DBL_EPSILON__){
        TTC = NAN;
    }
    else{
        TTC = medianXCurr*dT / (medianXPrev - medianXCurr);
        if(TTC <= -20 || TTC >= 20){TTC = NAN;}
    }
}

//  输出当前index所在的bbox中的matchingBoxId
inline const bool matchPointToBoundingBox(const DataFrame &frame, const int index, int &matchingBoxId) {
	int matchCount{ 0 };
    const cv::Point2f point(frame.keypoints.at(index).pt);

    for (int currentBoundingBoxIndex{ 0 }; currentBoundingBoxIndex < frame.boundingBoxes.size(); currentBoundingBoxIndex++) {
        if (!frame.boundingBoxes.at(currentBoundingBoxIndex).roi.contains(point)) { continue; }

        if (++matchCount > 1) {  // 如果匹配点在两个以上的bbox中，则舍弃
            matchingBoxId = -1;
            return false;
        }

        matchingBoxId = currentBoundingBoxIndex;
    }

	return static_cast<bool>(matchCount);
}

inline const int getMaximumIndex(const std::vector<int> &inputData){
    using std::begin;
    using std::end;

	return std::distance(begin(inputData), std::max_element(begin(inputData), end(inputData)));
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
        DataFrame &prevFrame, DataFrame &currFrame) {

	const int columns{ static_cast<int>(prevFrame.boundingBoxes.size()) };
	const int rows{ static_cast<int>(currFrame.boundingBoxes.size()) };

    std::vector<std::vector<int>> listOfMatches(columns, std::vector<int>(rows, 0));

    int previousFrameMatchingBoundingBoxId{ -1 }, currentFrameMatchingBoundingBoxId{ -1 };

	for(auto match : matches){
		if(!matchPointToBoundingBox(prevFrame, match.queryIdx, previousFrameMatchingBoundingBoxId)) { continue; }
		if(!matchPointToBoundingBox(currFrame, match.trainIdx, currentFrameMatchingBoundingBoxId)) { continue; }

        ++listOfMatches.at(previousFrameMatchingBoundingBoxId).at(currentFrameMatchingBoundingBoxId);
	}

    for(int i =0; i < listOfMatches.size(); i ++) {
        for (int j = 0; j < listOfMatches[i].size(); j++) {
            std::cout << listOfMatches[i][j] << " ";
        }
        std::cout << std::endl;
    }

    for (int columnIndex{ 0 }; columnIndex < prevFrame.boundingBoxes.size(); columnIndex++) {
        const int rowIndex{ getMaximumIndex(listOfMatches.at(columnIndex)) };

        if (listOfMatches.at(columnIndex).at(rowIndex) == 0) { continue; }

        bbBestMatches[prevFrame.boundingBoxes.at(columnIndex).boxID] = currFrame.boundingBoxes.at(rowIndex).boxID;
    }
}
