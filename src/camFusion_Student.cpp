
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <math.h>
#include <bits/stdc++.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Exclude keypoints that do not belong to exactly one bounding box
std::vector<cv::KeyPoint> eraseKptsInOverlappingBBs(std::vector<BoundingBox> &boundingBoxes,
                                                    std::vector<cv::KeyPoint> &keypoints)
{
    std::vector<cv::KeyPoint> filteredKeypoints;
    // loop over all keypoints and associate them to a 2D bounding box
    for (auto currKpt : keypoints)
    {
        int nBBsContainingKpt = 0;
        for (auto it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = it2->roi.x;
            smallerBox.y = it2->roi.y;
            smallerBox.width = it2->roi.width;
            smallerBox.height = it2->roi.height;

            // check whether point is within current bounding box
            if (smallerBox.contains(currKpt.pt))
                nBBsContainingKpt++;

        } // end loop over all bounding boxes

        // in order to avoid inadvertent association of a keypoint on one vehicle with another vehicle,
        // exclude keypoints that are enclosed within multiple bounding boxes from further processing
        if (nBBsContainingKpt == 1) // if keypoint is enclosed by exactly one bounding box
            filteredKeypoints.push_back(currKpt);

    } // end loop over all keypoints
    return filteredKeypoints;
}

// Create groups of lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // convert current lidar point into homogeneous coordinates for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project lidar point X into the image plane of the camera
        Y = P_rect_xx * R_rect_xx * RT * X;

        // transform Y back into Euclidean coordinates and store the result in pt
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // set pointers to all bounding boxes which enclose the current LIDAR point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box by shrinkFactor to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check whether the point falls within the current bounding box
            if (smallerBox.contains(pt))
                enclosingBoxes.push_back(it2);

        } // end loop over all bounding boxes

        // in order to avoid inadvertent association of a lidar point on one vehicle with another vehicle,
        // exclude lidar points that are enclosed within multiple bounding boxes from further processing
        if (enclosingBoxes.size() == 1)                     // if lidar point is enclosed by only one bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1); // then add the lidar point to the bounding box

    } // end loop over all lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0)); //cv::Scalar(255, 255, 255));

    // plot distance markers: add a horizontal line every 0.5m
    float lineSpacing = 0.5; // gap between distance markers
  
    // compute the number of markers, scaled by the height of the top view image
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot lidar points into top view image
        float top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        float red, green;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            red = 255 * abs((bottom - y) / (bottom - top));         //red = 255 * abs((1318.0-y) / (1318.0-1159.0));
            green = 255 * (1 - abs((bottom - y) / (bottom - top))); //green = 255 * (1 - abs((1318.0-y) / (1318-1159)));
            cv::circle(topviewImg, cv::Point(x, y), 1, cv::Scalar(0, green, red), 1.8);
            //cv::circle(topviewImg, cv::Point(x, y), 1, currColor, 1.8);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point2f(left - 1, top - 1), cv::Point2f(right + 1, bottom + 1), cv::Scalar(255, 255, 255), 0.1);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 0, bottom + 15), cv::FONT_ITALIC, 0.5, cv::Scalar(255, 255, 255), 1.2, 8);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 0, bottom + 30), cv::FONT_ITALIC, 0.5, cv::Scalar(0, 255, 0), 1.2, 8);
    }

    // crop image
    cv::Rect roi(690, 1155, 795, 200); // top left x, top left y, width, height
    topviewImg = topviewImg(roi);

    if (bWait)
    {
        // display image
        string windowName = "3D Objects";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL); //1);
        cv::imshow(windowName, topviewImg);
        cv::waitKey(0); // wait for key to be pressed
    }
    //bool check = imwrite(imgMatchFile,topviewImg);
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBoxPrev, BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches)
{
    // init variables for mean-based match selection
    double meanDist = 0.0, sumDist = 0.0;
    int numGoodMatches = 0;
    std::map<cv::DMatch, double> bbMatches;
    std::vector<cv::KeyPoint> tmpKpts, tmpKptsPrev;

    // init variables to find good matches using RANSAC
    std::vector<cv::Point2f> pointsPrev, pointsCurr;
    std::vector<cv::DMatch> bbMatches2;

    for (auto match : kptMatches)
    {
        // shrink current bounding box slightly to avoid having too many outlier points around the edges
        cv::Rect smallerBox;
        smallerBox.x = boundingBox.roi.x + 0.05 * boundingBox.roi.width / 2.0;
        smallerBox.y = boundingBox.roi.y + 0.05 * boundingBox.roi.height / 2.0;
        smallerBox.width = boundingBox.roi.width * (1 - 0.01);
        smallerBox.height = boundingBox.roi.height * (1 - 0.2);

        // if the smaller bounding box contains the current keypoint in the current match
        // note: kptMatches are ordered as follows: kptsPrev uses queryIdx and kptsCurr uses trainIdx
        if (smallerBox.contains(kptsCurr[match.trainIdx].pt))
        {
            // convert keypoints to Point2f (for RANSAC)
            pointsPrev.push_back(cv::Point2f(kptsPrev[match.queryIdx].pt.x, kptsPrev[match.queryIdx].pt.y));
            pointsCurr.push_back(cv::Point2f(kptsCurr[match.trainIdx].pt.x, kptsCurr[match.trainIdx].pt.y));

            // save keypoints within the bounding box
            tmpKpts.push_back(kptsCurr[match.trainIdx]);
            tmpKptsPrev.push_back(kptsPrev[match.queryIdx]);

            // compute the Euclidean distance between the previous and current positions of the keypoint
            double distance = cv::norm(kptsCurr[match.trainIdx].pt - kptsPrev[match.queryIdx].pt);
            bbMatches[match] = distance; // save distance to match structure
            bbMatches2.push_back(match);
            sumDist += distance; // keep a running sum of distances between mapped keypoint positions
            numGoodMatches++;    // keep a counter of number of good keypoint matches
        }
    }

    bool useRansac = true;
    if (useRansac) // Method 1: Use RANSAC to find inlier matches
    {
        // Compute the homography matrix and inlier (good) matches using RANSAC
        std::vector<uchar> inliers; //cv::Mat inliers;
        cv::Mat H = cv::findHomography(
            cv::Mat(pointsPrev), // matching points in previous frame
            cv::Mat(pointsCurr), // matching points in current frame
            cv::RANSAC,          // RANSAC method
            3,                   // RANSAC reprojection threshold
            inliers);            // output mask where 0s indicate outliers

        // select only the inliers
        for (int i = 0; i < inliers.size(); i++)
        {
            if ((int)inliers.at(i))
            {
                boundingBox.kptMatches.push_back(bbMatches2[i]);
                boundingBoxPrev.kptMatches.push_back(bbMatches2[i]);

                boundingBox.keypoints.push_back(tmpKpts[i]);
                boundingBoxPrev.keypoints.push_back(tmpKptsPrev[i]);
            }
        }
    }
    else // Method 2: Apply a threshold multiplier to the mean match distance to filter outliers
    {
        // filter out matches that are not within a threshold of the mean match distance
        int kptNum = 0;
        float outlierThresh = 2.5;           // empirically selected threshold
        meanDist = sumDist / numGoodMatches; // compute the average distance
        for (auto matchDist : bbMatches)     // loop over all bounding box matches
        {
            if (matchDist.second < outlierThresh * meanDist)
            {
                boundingBox.kptMatches.push_back(matchDist.first);
                boundingBoxPrev.kptMatches.push_back(matchDist.first);

                boundingBox.keypoints.push_back(tmpKpts[kptNum]);
                boundingBoxPrev.keypoints.push_back(tmpKptsPrev[kptNum]);
            }
            kptNum++;
        }
    }
}

// Define helper function to recursively generate all pairwise combinations of keypoints
void comboHelper(vector<vector<int>> &ans, vector<int> &tmp, int n, int left, int k)
{
    // if done, return result
    if (k == 0)
    {
        ans.push_back(tmp);
        return;
    }

    for (int i = left; i <= n; ++i)
    {
        tmp.push_back(i);
        comboHelper(ans, tmp, n, i + 1, k - 1);
        tmp.pop_back();
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
// Note: I used the code provided by Andreas Haja in the classroom exercise
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, size_t imgIndex, string filename) //cv::Mat *visImg)
{
    // compute distance ratios for all matched keypoints between the current and previous frame
    vector<double> distRatios;

    // compute the distance between every keypoint match pair with every other keypoint match pair
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        // get current keypoint and its matched partner in the previous frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0; // exclude identical or very close keypoints

            // get next keypoint and its matched partner in the previous frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute Euclidean distances between pairs of keypoint matches
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
          
            /*if (distCurr < minDist)
            {
              //std::cout << "Discarded Match Pair: outer=[" << kpOuterCurr.pt.x << "," << kpOuterCurr.pt.y << "], inner=[" << kpInnerCurr.pt.x << "," << kpInnerCurr.pt.y << "], dist=" << distCurr << std::endl;
              ofstream logfile("Discarded_Keypoints.csv", ios_base::app);
                    logfile << imgIndex << ", " << kpOuterCurr.pt.x << ", " << kpOuterCurr.pt.y
                            << ", " << kpInnerCurr.pt.x << ", " << kpInnerCurr.pt.y
                            << ", " << distCurr << "\n";
                    logfile.close();
            }*/

            // prevent divide by zero
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);

                bool printCameraPoints = false;
                if (printCameraPoints)
                {
                    ofstream logfile(filename, ios_base::app);
                    logfile << imgIndex << ", " << kpOuterCurr.pt.x << ", " << kpOuterCurr.pt.y
                            << ", " << kpOuterPrev.pt.x << ", " << kpOuterPrev.pt.y
                            << ", " << kpInnerCurr.pt.x << ", " << kpInnerCurr.pt.y
                            << ", " << kpInnerPrev.pt.x << ", " << kpInnerPrev.pt.y
                            << ", " << distCurr << ", " << distPrev << ", " << distRatio << "\n";
                    logfile.close();
                  
                                /**********************************************************************************************************
                                // store matches in current data frame
                                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                                currBB->kptMatches, matchImg,
                                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                                vector<char>()); //, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); //DRAW_RICH_KEYPOINTS);
                                string windowName = "Matching keypoints between two camera images"; cv::namedWindow(windowName, 7);
                                cv::imshow(windowName, matchImg); cv::waitKey(0);
                                bool check = imwrite(imgMatchFile, matchImg);
                                **********************************************************************************************************/
                }
            }
        } // end inner loop over all matched keypoints
    }     // end outer loop over all matched keypoints

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute median distance ratio, removing outlier influence
    std::sort(distRatios.begin(), distRatios.end());

    long medianIdx = floor(distRatios.size() / 2.0);
    double medianDistRatio;
    if (distRatios.size() % 2 != 0) // if the number of distance ratio elements is odd
    {
        // then take the middle value
        medianDistRatio = distRatios[medianIdx];
    }
    else // if the number of distance ratio elements is even
    {
        // then take the average of the two middle values
        medianDistRatio = (distRatios[medianIdx - 1] + distRatios[medianIdx]) / 2;
    }

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);

    std::cout << "Camera: TTC = " << TTC << ", Median Dist Ratio = " << medianDistRatio << std::endl;
    //ofstream logfile(filename, ios_base::app);
    //logfile << imgIndex << ", " << TTC << ", " << medianDistRatio << std::endl;
    //logfile.close();
}

// Compute the time-to-collision for all matched 3D objects based on lidar measurements alone
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // reformat x (distance) values as a vector of doubles
    std::vector<double> distPrev;
    std::vector<double> distCurr;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
        distPrev.push_back(it->x);

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
        distCurr.push_back(it->x);

    // take nth smallest point as the distance to use (anything smaller is discarded as noise)
    int nthP, nthC;
    float prctile = 0.25; //0.5; // nth point is based on a fraction of the total number of points
    nthP = floor(prctile * distPrev.size());
    std::nth_element(distPrev.begin(), distPrev.begin() + (nthP - 1), distPrev.end());
    nthC = floor(prctile * distCurr.size());
    std::nth_element(distCurr.begin(), distCurr.begin() + (nthC - 1), distCurr.end());

    // based on the model of constant velocity, the TTC can be computed from two successive lidar measurements as follows:
    TTC = (distCurr[nthC - 1] * (1 / frameRate)) / (distPrev[nthP - 1] - distCurr[nthC - 1]);
    //TTC = (d_curr*(1/frameRate))/(d_prev-d_curr);

    std::cout << "LIDAR: TTC = " << TTC << ", Prev Dist = " << distPrev[nthP - 1] << ", Curr Dist = " << distCurr[nthC - 1] << std::endl;
}

// Associate bounding boxes across frames based on the highest number of keypoint matches
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // count the number of keypoint matches that share the same bounding box between the current and previous frames
    cv::Mat count = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32S);
    for (auto matchpair : matches) // for each matched pair ...
    {
        // find the corresponding keypoint in the current and previous frame
        // note: kptMatches are ordered as follows: kptsPrev uses queryIdx and kptsCurr uses trainIdx
        cv::KeyPoint prevKp = prevFrame.keypoints.at(matchpair.queryIdx);
        cv::KeyPoint currKp = currFrame.keypoints.at(matchpair.trainIdx);

        // for each bounding box on the previous frame
        for (int prevbb = 0; prevbb < prevFrame.boundingBoxes.size(); prevbb++)
        {
            // if the bounding box on the previous frame contains this keypoint
            if (prevFrame.boundingBoxes[prevbb].roi.contains(prevKp.pt))
            {
                // then loop through all bounding boxes on the current frame
                for (int currbb = 0; currbb < currFrame.boundingBoxes.size(); currbb++)
                {
                    // if bounding box on the current frame also contains this keypoint
                    if (currFrame.boundingBoxes[currbb].roi.contains(currKp.pt))
                    {
                        // then increment multimap counter for matches that share the same bounding box
                        count.at<int>(prevbb, currbb) = count.at<int>(prevbb, currbb) + 1;
                    }
                }
            }
        }
    } // end loop over all matched pairs between frames

    // associate bounding boxes that contain the highest number of matched pairs
    for (int prevBBIdx = 0; prevBBIdx < prevFrame.boundingBoxes.size(); prevBBIdx++)
    { // for each bounding box on the previous frame
        int boxID = -1, maxNumMatches = 0;
        for (int currBBIdx = 0; currBBIdx < currFrame.boundingBoxes.size(); currBBIdx++)
        { // for each bounding box on the current frame
            if (count.at<int>(prevBBIdx, currBBIdx) > maxNumMatches)
            { // find which pair of bounding boxes on prev and current frames have the maximum count
                boxID = currBBIdx;
                maxNumMatches = count.at<int>(prevBBIdx, currBBIdx);
            }
        }
        // save the box IDs of all matched pairs in bbBestMatches
        bbBestMatches[prevBBIdx] = boxID;
    }
}
