
#include <numeric>
#include "matching2D.hpp"

using namespace std;

/* +---------------------------+ */
/* | KEYPOINT MATCHING METHODS | */
/* +---------------------------+ */
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // Match descriptor vectors between adjacent video frames using Brute Force or FLANN
    // Note that some points will not be assigned a match
    bool crossCheck = false; // alternative to the distance ratio test
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // Configure & Create the Matcher Object
    if (matcherType.compare("BF_MATCH") == 0) // Brute Force Matching
    {
        // Set optimal distance measure for efficient feature matching:
        // Binary string descriptors: BRIEF, BRISK, FREAK, ORB, AKAZE, etc.
        // Floating point descriptors: SIFT, SURF, GLOH, etc.
        int normType;        
        if (descriptorType.compare("BRIEF") == 0 ||
            descriptorType.compare("BRISK") == 0 ||
            descriptorType.compare("ORB")   == 0 ||
            descriptorType.compare("FREAK") == 0 ||
            descriptorType.compare("AKAZE") == 0)
        { // if matching binary string-based descriptors
            // then count the number of different elements for binary strings (for efficiency)
            normType = cv::NORM_HAMMING;
        }
        else // (descriptorType.compare("SIFT") == 0)
        { // matching HOG descriptors like SIFT, SURF, GLOH, etc
            // then use the Euclidean distance (L2-norm) for gradient-based, floating point descriptors
            normType = cv::NORM_L2; // (default)
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("FLANN_MATCH") == 0) // Fast Library for Approximate Nearest Neighbors
    {
        if (descRef.type() != CV_32F || descSource.type() != CV_32F)
        { // FLANN needs the descriptors to be of type CV_32F
            descRef.convertTo(descRef, CV_32F);
            descSource.convertTo(descSource, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    else
    {
        std::cerr << "ERROR: Matcher Type " << matcherType << "Not Recognized!" << std::endl;
        return;
    }

    // Select Best Matches
    if (selectorType.compare("KNN") == 0)
    { 
        // k nearest neighbors: keep the best k matches
        vector<vector<cv::DMatch>> knn_matches;
        int numNeighbors = 2; // find the 2 best matches
        matcher->knnMatch(descSource, descRef, knn_matches, numNeighbors);

        // Apply the distance ratio test to try to filter out false matches (alternative to crossCheck)
        //  For a given keypoint:
        //    - take the ratio of the distances between the closest neighbor and the second closest neighbor 
        //    - compare the distance ratio to a threshold of minDescDistRatio
        //  If the ratio is above threshold, then the match is ambiguous with a lower probability of being correct
        //  If the ratio is below threshold, then accept the closest match
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ( (*it)[0].distance < minDescDistRatio * (*it)[1].distance ) // use multiplication to increase clock speed
            {
                // keep the most likely matches
                matches.push_back((*it)[0]);
            }
        }
    }
    else // (selectorType.compare("NN") == 0)
    { 
        // nearest neighbor: keep only the best match (default setting)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
}

/* +-----------------------------+ */
/* | KEYPOINT DESCRIPTOR METHODS | */
/* +-----------------------------+ */
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // Select appropriate descriptor. Valid Options are: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cerr << "ERROR: Descriptor Type " << descriptorType << "Not Recognized!" << std::endl;
        return;
    }
    
    // perform feature description
    extractor->compute(img, keypoints, descriptors); // compute descriptors
}

/* +----------------------------+ */
/* | KEYPOINT DETECTION METHODS | */
/* +----------------------------+ */
void detKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType)
{
    // Valid options are: HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    if (detectorType.compare("SHITOMASI") == 0) // Shi-Tomasi detector
    {
        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
        double qualityLevel = 0.01; // minimal accepted quality of image corners
        double k = 0.04;
        vector<cv::Point2f> corners;

        // Apply corner detection
        cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

        // Add corners to result vector
        for (auto it = corners.begin(); it != corners.end(); ++it)
        {
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
            newKeyPoint.size = blockSize;
            keypoints.push_back(newKeyPoint);
        }   
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        // Detector parameters
        int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
        int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
        int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
        double k = 0.04;       // Harris parameter (see equation for details)

        // Detect Harris corners and normalize output
        cv::Mat dst, dst_norm, dst_norm_scaled;
        dst = cv::Mat::zeros(img.size(), CV_32FC1);

        // run the Harris edge detector on the image
        cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        //cv::convertScaleAbs(dst_norm, dst_norm_scaled); // removed to facilitate run-time only evaluation

        // Look for prominent corners
        double maxOverlap = 0.0; // max permissible overlap between two features in %, used during non-maxima suppression
        for (size_t j = 0; j < dst_norm.rows; j++)
        {
            for (size_t i = 0; i < dst_norm.cols; i++)
            {
                int response = (int)dst_norm.at<float>(j, i);
                if (response > minResponse) // only store points above a threshold
                {       
                    cv::KeyPoint newKeyPoint;
                    newKeyPoint.pt = cv::Point2f(i, j);
                    newKeyPoint.size = 2 * apertureSize;
                    newKeyPoint.response = response;

                    // perform non-maximum suppression (NMS) in local neighborhood around new key point
                    bool bOverlap = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                        if (kptOverlap > maxOverlap)
                        {
                            bOverlap = true;
                            if (newKeyPoint.response > (*it).response)
                            {                      // if overlap is >t AND response is higher for new kpt
                                *it = newKeyPoint; // replace old key point with new one
                                break;             // quit loop over keypoints
                            }
                        }
                    }
                    // only add new key point if no overlap has been found in previous NMS
                    if (!bOverlap)
                        keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list

                } // end if points above threshold
            } // end loop over cols
        } // end loop over rows

    }
    else // FAST, BRISK, ORB, AKAZE, SIFT --> Use OpenCV Library 
    {
        cv::Ptr<cv::FeatureDetector> detector;
        if (detectorType.compare("FAST") == 0)
        {   
            int threshold = 60; //30; // difference between intensity of the central pixel and pixels of a circle around this pixel
            bool bNMS = true;  // perform non-maxima suppression on keypoints
            cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_7_12; // TYPE_9_16, TYPE_7_12, TYPE_5_8
            detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        }
        else if (detectorType.compare("BRISK") == 0)
        {
            detector = cv::BRISK::create(); // this constructor takes ~330 ms (detect below only takes 5)!!!
        }
        else if (detectorType.compare("ORB") == 0)
        {
            detector = cv::ORB::create();
        }
        else if (detectorType.compare("AKAZE") == 0)
        {
            detector = cv::AKAZE::create();
        }
        else if (detectorType.compare("SIFT") == 0)
        {
            detector = cv::xfeatures2d::SIFT::create();
        }
        else
        {
            std::cerr << "ERROR: Detector Type " << detectorType << "Not Recognized!" << std::endl;
            return;
        }
    
        // detect keypoints
        detector->detect(img, keypoints); // detect keypoints                
    }
}