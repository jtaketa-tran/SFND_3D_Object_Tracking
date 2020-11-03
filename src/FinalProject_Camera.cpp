/* +---------------------------+ */
/* | INCLUDES FOR THIS PROJECT | */
/* +---------------------------+ */
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
#include <boost/circular_buffer.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

void loadCalibrationData(cv::Mat &P_rect_00, cv::Mat &R_rect_00, cv::Mat &RT)
{
    // numbers from the KITTI camera calibration files specific to the sensor setup in KITTI sequence we are using
    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;
}

/* +--------------+ */
/* | MAIN PROGRAM | */
/* +--------------+ */
int main(int argc, const char *argv[])
{
    /* +------------------------------------+ */
    /* | INIT VARIABLES AND DATA STRUCTURES | */
    /* +------------------------------------+ */
    // keypoint detectors and descriptors
    const int NUM_DET_TYPES = 7;
    const int NUM_DESC_TYPES = 6;
    const char *DetectorTypes[NUM_DET_TYPES] = {"SIFT", "FAST", "BRISK", "ORB", "AKAZE", "SHITOMASI", "HARRIS"};
    const char *DescriptorTypes[NUM_DESC_TYPES] = {"SIFT", "AKAZE", "BRIEF", "ORB", "FREAK", "BRISK"};

    // matcher and selector
    string matcherType = "BF_MATCH"; // BF_MATCH, FLANN_MATCH
    string selectorType = "KNN";     // NN, KNN

    // keypoint viewing options
    bool visKeypoints = false;       // run routine to visualize and/or save keypoints overlaid on car cookie cuts without option to writeImages
    bool visMatches = false;         // run routine to visualize and/or save adjacent frame keypoint matches with option to writeImages
    bool writeImages = false;        // save images with keypoint and/or match overlays to files
    bool plotImages = false;         // visualize images with keypoint and/or match overlays during run time
    bool cropImage4PlotOnly = false; // for plotting keypoints (not actually cropping the image)
    bool bFocusOnVehicle = false;    // shortcut for this study only: discard any keypoints outside a bounding box on the preceding vehicle
    bool bVis = false;               // visualize results (generic flag used elsewhere)
    string kptImgFilename = "Keypoints_";
    string matchImgFilename = "Matches_";

    // object detection
    string dataPath = "../"; // data location
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // KITTI camera images
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color version
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;  // last file index to load
    int imgStepWidth = 1;  // option to downsample the frame rate
    int imgFillWidth = 4;  // number of digits in file index (e.g. img-0001.png)
    
    // KITTI LIDAR data
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // KITTI calibration data for camera and lidar
    cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 intrinsic camera calibration / projection matrix after rectification
    cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make left and right stereo camera images in the KITTI video co-planar
    cv::Mat RT(4, 4, cv::DataType<double>::type);        // extrinsic rotation matrix and translation vector
    loadCalibrationData(P_rect_00, R_rect_00, RT);

    // other params
    double sensorFrameRate = 10.0 / imgStepWidth; // 10Hz scaled by the image step width; frames per second for Lidar and camera
    int dataBufferSize = 2;                       // number of consecutive images that will be held in memory (ring buffer)

  	// loop over keypoint detector types
    for (int detIdx = 0; detIdx < NUM_DET_TYPES; detIdx++)
    {
        string detectorType = DetectorTypes[detIdx];
        int firstTime = 1; // so that we only plot keypoints once
        int firstTimeCurrFrame = 1;
      
        // loop over keypoint descriptor types
        for (int descIdx = 0; descIdx < NUM_DESC_TYPES; descIdx++)
        {
            string descriptorType = DescriptorTypes[descIdx];

            // skip incompatible combinations of keypoint detectors and descriptors:
            // - AKAZE descriptors can only be computed on KAZE or AKAZE detectors
            if (descriptorType == "AKAZE" && detectorType != "AKAZE")
                continue;
            // - ORB descriptors cannot be computed on SIFT detectors
            if (descriptorType == "ORB" && detectorType == "SIFT")
                continue;

            cout << "detectorType: " << detectorType << ", descriptorType: " << descriptorType << endl;

            // create ring buffer for video frames
            boost::circular_buffer<DataFrame> dataBuffer;
            dataBuffer.set_capacity(dataBufferSize);

            /* +---------------------------+ */
            /* | MAIN LOOP OVER ALL IMAGES | */
            /* +---------------------------+ */
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth)
            {
                /* +---------------------------------------------------------+ */
                /* | LOAD CAMERA IMAGE AND INSERT INTO 2-ELEMENT RING BUFFER | */
                /* +---------------------------------------------------------+ */
                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // assemble filenames for saving images
                ostringstream imgIdx;
                imgIdx << imgStartIndex + imgIndex;
                string delim = "_";
                string imgKpFile = kptImgFilename + detectorType + delim + imgIdx.str() + imgFileType;
                string imgMatchFile = matchImgFilename + detectorType + delim + descriptorType + delim +
                                      matcherType + delim + selectorType + delim + imgIdx.str() + imgFileType;

                // load color image from file
                cv::Mat img = cv::imread(imgFullFilename);

                // push images into ring buffer of size dataBufferSize = 2
                DataFrame frame; frame.cameraImg = img;
                dataBuffer.push_back(frame);

                cout << "PROCESSING IMAGE " << imgIndex << endl;

                /* +---------------------------------------------------------+ */
                /* | DETECT & CLASSIFY OBJECTS IN 2D COLOR IMAGES USING YOLO | */
                /* +---------------------------------------------------------+ */
                float confThreshold = 0.2;
                float nmsThreshold = 0.4;
                string yoloFilename = "YoloBBs" + imgIdx.str() + imgFileType;
                //bVis = true; // uncomment to plot camera images with YOLO bounding boxes
                detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                              yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis, yoloFilename);
                //bVis = false; continue;

                /* +----------------------------------------------+ */
                /* | ASSOCIATE LIDAR POINTS WITH CAMERA-BASED ROI | */
                /* +----------------------------------------------+ */
                // load 3D Lidar points from file
                string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
                std::vector<LidarPoint> lidarPoints;
                loadLidarFromFile(lidarPoints, lidarFullFilename);

                // crop Lidar points to constrain the 3D space for TTC computation
                float minZ = -1.6;  // slightly above road level, assumes a level road surface (no significant incline)
                float maxZ = -0.5;  //-0.9 // exclude points above roof of ego car
                float minX = 0.5;   //0.5 exclude points behind lidar sensor and immediately in front
                float maxX = 9.0;  //20 // exclude points too far away (upper distance limit)
                float maxY = 2.0;   // focus on ego lane for collision detection
                float minR = 0.001; //0.1; // reflectivity close to zero might indicate low reliability
                cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
              
                /*
              	ofstream logfile2("LIDAR_front.csv", ios_base::app);
                for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
                	logfile2 << imgIndex << ", " << it->x << ", " << it->y << ", " << it->z << ", " << it->r << ", " << "\n";
                logfile2.close();
                continue;
                */

                (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

                // cluster Lidar points by associating them with the camera-based ROI
                float shrinkFactor = 0.1; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
                clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

                // Visualize 3D objects: uncomment below to generate top-down LIDAR data on preceding car
                //show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);

                /* +------------------------+ */
                /* | DETECT IMAGE KEYPOINTS | */
                /* +------------------------+ */
                // convert current image to grayscale
                cv::Mat imgGray;
                cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints;
                detKeypoints(keypoints, imgGray, detectorType);
              
                // only keep keypoints on the preceding vehicle (not used in this project)
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    vector<cv::KeyPoint> vehicleKeypoints;
                    for (auto kp : keypoints)
                    {
                        if (vehicleRect.contains(kp.pt))
                            vehicleKeypoints.push_back(kp);
                    }
                    keypoints = vehicleKeypoints;
                }

                // exclude keypoints outside of any bounding boxes and in multiple bounding boxes
                std::vector<cv::KeyPoint> filteredKeypoints =
                    eraseKptsInOverlappingBBs((dataBuffer.end() - 1)->boundingBoxes, keypoints);

                // visualize keypoint detections
                if (visKeypoints && firstTime)
                {
                    firstTime == 0; // don't plot keypoints again
                    cv::Mat visImage = imgGray.clone();
                    cv::Mat visImageFiltered = imgGray.clone();
                    cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    cv::drawKeypoints(imgGray, filteredKeypoints, visImageFiltered, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    if (cropImage4PlotOnly)
                    {
                        cv::Rect roi(505, 150, 245, 225.5); // top left x, top left y, width, height
                        visImage = visImage(roi);
                        visImageFiltered = visImageFiltered(roi);
                    }

                    if (plotImages)
                    {
                        string windowName = detectorType;
                        cv::namedWindow(windowName, 6);
                        imshow(windowName, visImage);
                        cv::waitKey(0);
                    }

                    if (writeImages)
                    {
                        bool check = imwrite(imgKpFile, visImage);
                        check = imwrite("filtered" + delim + imgKpFile, visImageFiltered);
                    }
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = filteredKeypoints;
                //(dataBuffer.end() - 1)->keypoints = keypoints;

                /* +------------------------------+ */
                /* | EXTRACT KEYPOINT DESCRIPTORS | */
                /* +------------------------------+ */
                cv::Mat descriptors;
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                /* +----------------------------+ */
                /* | MATCH KEYPOINT DESCRIPTORS | */
                /* +----------------------------+ */
                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    vector<cv::DMatch> matches;
                    // note: matches are ordered from current to previous
                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                     (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                     matches, descriptorType, matcherType, selectorType);

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    // visualize matches between current and previous image
                    if (visMatches)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                        if (plotImages)
                        {
                            string windowName = "Matching keypoints between two camera images";
                            cv::namedWindow(windowName, 7);
                            cv::imshow(windowName, matchImg);
                            cv::waitKey(0); // wait for key to be pressed
                        }

                        if (writeImages)
                            bool check = imwrite(imgMatchFile, matchImg);
                    }

                    /* +---------------------------------------------------------+ */
                    /* | MATCH BOUNDING BOXES BETWEEN CURRENT AND PREVIOUS FRAME | */
                    /* +---------------------------------------------------------+ */
                    // associate bounding boxes between current and previous frame using keypoint matches
                    map<int, int> bbBestMatches;
                    matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1));
                    
                    // store best bounding box matches in current frame to the data buffer
                    (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

                    /* +------------------------------------+ */
                    /* | COMPUTE TTC TO THE OBJECT IN FRONT | */
                    /* +------------------------------------+ */
                    // loop over all bounding box matches identified in matchBoundingBoxes
                    for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                    {
                        BoundingBox *prevBB, *currBB;
                      
                        // find bounding boxes associated with the current match
                        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->second == it2->boxID) // if the current match partner corresponds to this bounding box
                                currBB = &(*it2);          // then get pointer to the bounding box in the current frame
                        }
                        
                        // find bounding boxes associated with the previous match
                        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->first == it2->boxID) // if the current match partner corresponds to this bounding box
                                prevBB = &(*it2);         // then get pointer to the bounding box in the previous frame
                        }

                        // if the bounding boxes on the curr and prev frames have lidar points associated with them
                        // (note: lidar points were previously cropped to capture obstacles only in the ego lane)
                        if (currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0)
                        {
                            double ttcLidar = -1;
                            
                            // then compute time-to-collision based on LIDAR data for the current match
                            computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);

                            // print lidar points to file
                            bool printLidarPoints = false;
                            if (printLidarPoints)
                            {
                                ofstream logfile("TTC_LIDAR_perc25.csv", ios_base::app);
                                if (imgIndex == 1)
                                { // then log information from imgIndex 0 first
                                    for (auto it = prevBB->lidarPoints.begin(); it != prevBB->lidarPoints.end(); ++it)
                                        logfile << imgIndex - 1 << ", " << it->x << ", " << it->y << ", " << it->z << ", " << it->r << ", " << ttcLidar << "\n";
                                }
                                for (auto it = currBB->lidarPoints.begin(); it != currBB->lidarPoints.end(); ++it)
                                    logfile << imgIndex << ", " << it->x << ", " << it->y << ", " << it->z << ", " << it->r << ", " << ttcLidar << "\n";

                                logfile.close();
                            }

                            // assign enclosed keypoint matches to the bounding box
                            clusterKptMatchesWithROI(*prevBB, *currBB,
                                                     (dataBuffer.end() - 2)->keypoints,
                                                     (dataBuffer.end() - 1)->keypoints,
                                                     (dataBuffer.end() - 1)->kptMatches);

                            // if the bounding boxes on the curr and prev frames have keypoint matches associated with them
                            if (currBB->kptMatches.size() > 0 && prevBB->kptMatches.size() > 0)
                            {
                                // compute time-to-collision based on 2D camera data
                                double ttcCamera;
                                string kptTTCfilename = "CameraTTC_Ransac_" + kptImgFilename + detectorType + delim + descriptorType + ".csv";
                                computeTTCCamera((dataBuffer.end() - 2)->keypoints,
                                                 (dataBuffer.end() - 1)->keypoints,
                                                 currBB->kptMatches,
                                                 sensorFrameRate,
                                                 ttcCamera,
                                                 imgIndex,
                                                 kptTTCfilename);

                                /**********************************************************************************************************
                                // draw lines between keypoints                   
                                if (imgIndex==1) // plot frame 0 first
                                {                                    
                                    cv::Mat visImgPrev = (dataBuffer.end() - 2)->cameraImg.clone();
                                    cv::Mat overlayPrev = visImgPrev.clone();
                          		    for (auto itPrev = prevBB->keypoints.begin(); itPrev != prevBB->keypoints.end()-1; ++itPrev)
                          		    {
                          			    for (auto itPrev2 = prevBB->keypoints.begin()+1; itPrev2 != prevBB->keypoints.end(); ++itPrev2)
                                            cv::line(overlayPrev, itPrev->pt, itPrev2->pt, cv::Scalar(0, 255, 0), 1);
                          		    }
                                    float opacity = 0.6; cv::addWeighted(overlayPrev, opacity, visImgPrev, 1 - opacity, 0, visImgPrev);
                                    cv::Rect prevroi(prevBB->roi.x, prevBB->roi.y, prevBB->roi.width, prevBB->roi.height); visImgPrev = visImgPrev(prevroi);
                                    //string windowName = "Relative Distance Between Keypoints"; cv::namedWindow(windowName, 5);
                                    //cv::imshow(windowName, visImgPrev); cv::waitKey(0);
                                    string imgKpFilePrev = "RelDist_Ransac_" + kptImgFilename + 
                                      detectorType + delim + descriptorType + delim + "0" + imgFileType;
                                    bool check = imwrite(imgKpFilePrev, visImgPrev);
                                }                                
                                // all other frames
                                cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                                cv::Mat overlay = visImg.clone();
                          		for (auto itCurr = currBB->keypoints.begin(); itCurr != currBB->keypoints.end()-1; ++itCurr)
                          		{
                          			for (auto itCurr2 = currBB->keypoints.begin()+1; itCurr2 != currBB->keypoints.end(); ++itCurr2)
                                        cv::line(overlay, itCurr->pt, itCurr2->pt, cv::Scalar(0, 255, 0), 1);
                          		}
                                float opacity = 0.6; cv::addWeighted(overlay, opacity, visImg, 1 - opacity, 0, visImg);
                                cv::Rect roi(currBB->roi.x, currBB->roi.y, currBB->roi.width, currBB->roi.height); visImg = visImg(roi);
                                //string windowName = "Relative Distance Between Keypoints"; cv::namedWindow(windowName, 4);
                                //cv::imshow(windowName, visImg); cv::waitKey(0);
                                string imgKpFileCurr = "RelDist_Ransac_" + kptImgFilename + detectorType + delim + 
                                  descriptorType + delim + imgIdx.str() + imgFileType;
                                bool check = imwrite(imgKpFileCurr, visImg);
                                **********************************************************************************************************/
                                /**********************************************************************************************************
                                // plot lidar
                                cv::Mat visImg2 = (dataBuffer.end() - 1)->cameraImg.clone();
                                showLidarImgOverlay(visImg2, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg2);
                               	cv::rectangle(visImg2, cv::Point(currBB->roi.x, currBB->roi.y),
                                              cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height),
                                              cv::Scalar(0, 255, 0), 1.5);
                                char str[200]; sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                                putText(visImg2, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                                //string windowName = "Final Results : TTC"; cv::namedWindow(windowName, 4); 
                                //cv::imshow(windowName, visImg2); cv::waitKey(0);
                                string imgTTC = "TTC_Ransac_" + kptImgFilename + detectorType + delim + imgIdx.str() + imgFileType;
                                bool check = imwrite(imgTTC, visImg2);
                                **********************************************************************************************************/
                            } // end if the current and previous frames have keypoint matches associated with them

                        } // end TTC computation

                    } // end loop over all bounding box matches

                } // if at least 2 images have been processed

            } // end loop over all images

        } // end loop over descriptor types

    } // end loop over detector types

    return 0;
}
