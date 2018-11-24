#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

#include "frontend/VisualFrontend.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
// #include <ocl.hpp>

int main( int argc, char** argv )
{
    if ( argc != 3 )
    {
        std::cout<< "Usage: feature_extraction img1 img2" << "\n";
        return 1;
    }

    bool is_gpu = true;
    bool is_display_all_line = false;
    bool is_display_all_point = false;
    bool is_display_grid = true;
    int grid_size = 10;
    int num_point_per_cell = 1;
    float feature_detector_threshold = 0.5;
    float cross_validation_threshold = 0.05;
    std::list<cv::Point2f> results;

    // Read two images.
    cv::Mat img_1_src = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2_src = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_out;
    cv::hconcat(img_1_src, img_2_src, img_out);
    int roi_width = (int)(img_1_src.cols / grid_size);
    int roi_height = (int)(img_1_src.rows / grid_size);
    std::chrono::duration<double> time_used;
    std::vector<cv::Point2f> prev_keypoints;
    std::vector<cv::Point2f> next_keypoints;
    PyrLKOpticalFlow tracker;

    // Loop through the grids and find the key points.
    for (int y = 0; y < grid_size; y++)
    {
        for (int x = 0; x < grid_size; x++)
        {
            // Prepare grid patches.
            cv::Rect roi(x * roi_width, y * roi_height, roi_width, roi_height);
            cv::Mat img_1 = img_1_src(roi);
            cv::Mat img_2 = img_2_src(roi);
            if (is_display_grid) 
            {
                cv::rectangle(img_out, roi, cv::Scalar(100, 100, 100));
                roi.x += img_2_src.size[1];
                cv::rectangle(img_out, roi, cv::Scalar(100, 100, 100));
            }

            std::list<cv::Point2f> list_key_points;
            std::vector<cv::KeyPoint> vector_key_points;
            
            // Prepare both key points.
            std::vector<cv::Point2f> keypoints_1;
            std::vector<cv::Point2f> keypoints_1_2;
            std::vector<cv::Point2f> keypoints_2_1;

            if (is_gpu)
            {
                // Detect features.
                GoodFeaturesToTrackDetector_GPU detector;
                detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);
                
                GpuMat d_img_1(img_1);
                GpuMat d_vector_key_points;
                
                detector(d_img_1, d_vector_key_points);
                VisualFrontend::downloadPoints(
                    d_vector_key_points, keypoints_1);
            }
            else
            {
                // Detect features.
                std::string detectorType = "Feature2D.BRISK";
                cv::Ptr<cv::FeatureDetector>detector = 
                    cv::Algorithm::create<cv::FeatureDetector>(detectorType);
                detector->set("thres", feature_detector_threshold);
                detector->detect(img_1, vector_key_points);
                
                // Convert keypoints points.
                for (auto kp : vector_key_points)
                {
                    keypoints_1.push_back(kp.pt);
                }
            }

            // Skip if no result found.
            if (keypoints_1.size() == 0) {continue;}

            // Prepare for optical flow matching.
            std::vector<unsigned char> status_1;
            std::vector<unsigned char> status_2;
            std::vector<float> error_1;
            std::vector<float> error_2;
            std::chrono::steady_clock::time_point t1;
            std::chrono::steady_clock::time_point t2;
            
            if (is_gpu)
            {
                cv::Mat mat_p = cv::Mat(1, keypoints_1.size(), CV_32FC2);
                for (size_t i = 0; i < keypoints_1.size(); i++)
                {
                    mat_p.at<cv::Vec2f>(0, i)[0] = keypoints_1[i].x;
                    mat_p.at<cv::Vec2f>(0, i)[1] = keypoints_1[i].y;
                }
                GpuMat d_img_1(img_1);
                GpuMat d_img_2(img_2);
                GpuMat d_keypoints_1(mat_p);
                GpuMat d_keypoints_1_2;
                GpuMat d_keypoints_2_1;         
                GpuMat d_status_1;
                GpuMat d_status_2;

                // Match the points and compute the time.
                t1 = std::chrono::steady_clock::now();
                
                tracker.sparse(
                    d_img_1, d_img_2, 
                    d_keypoints_1, d_keypoints_1_2, 
                    d_status_1);
                
                t2 = std::chrono::steady_clock::now();
                time_used += std::chrono::duration_cast
                    <std::chrono::duration<double, std::milli>>(t2 - t1);                  

                // Cross validate and check status.
                t1 = std::chrono::steady_clock::now();
                tracker.sparse(
                    d_img_2, d_img_1, 
                    d_keypoints_1_2, d_keypoints_2_1, 
                    d_status_2);
                t2 = std::chrono::steady_clock::now();
                time_used += std::chrono::duration_cast
                    <std::chrono::duration<double, std::milli>>(t2 - t1);                  

                // Download results to CPU.
                VisualFrontend::downloadPoints(d_keypoints_1, keypoints_1);
                VisualFrontend::downloadPoints(d_keypoints_1_2, keypoints_1_2);
                VisualFrontend::downloadPoints(d_keypoints_2_1, keypoints_2_1);
                VisualFrontend::downloadMasks(d_status_1, status_1);
                VisualFrontend::downloadMasks(d_status_2, status_2);
            }
            else
            {
                // Match the points and compute the time.
                t1 = std::chrono::steady_clock::now();
                                
                cv::calcOpticalFlowPyrLK(
                    img_1, img_2, 
                    keypoints_1, keypoints_1_2, 
                    status_1, error_1);
                t2 = std::chrono::steady_clock::now();
                time_used += std::chrono::duration_cast
                    <std::chrono::duration<double>>(t2 - t1);
                
                // Cross validate and check status.
                t1 = std::chrono::steady_clock::now();
                cv::calcOpticalFlowPyrLK(
                    img_2, img_1, 
                    keypoints_1_2, keypoints_2_1, 
                    status_2, error_2);
                t2 = std::chrono::steady_clock::now();
                time_used += std::chrono::duration_cast
                    <std::chrono::duration<double>>(t2 - t1);
            }

            int found = 0;
            for (size_t i = 0; i < status_1.size(); i++) 
            {
                if (status_1.at(i) == 1 && status_2.at(i) == 1)
                {
                    float distance = std::sqrt(
                        std::pow(keypoints_1[i].x - keypoints_2_1[i].x, 2) + 
                        std::pow(keypoints_1[i].y - keypoints_2_1[i].y, 2));
                    if (distance < cross_validation_threshold && 
                        found < num_point_per_cell)
                    {
                        // Get the first proper match. Per cell.
                        cv::Point2f p = keypoints_1.at(i);
                        p.x += x * roi_width;
                        p.y += y * roi_height;
                        prev_keypoints.push_back(p);

                        p = keypoints_1_2.at(i);
                        p.x += x * roi_width;
                        p.y += y * roi_height;
                        next_keypoints.push_back(p);
                        
                        found++;
                    } 
                    else
                    {
                        // Plot the points are filtered out by the validation.
                        cv::Point2f p1 = keypoints_1.at(i);
                        p1.x += x * roi_width;
                        p1.y += y * roi_height;
                        
                        cv::Point2f p2 = keypoints_1_2.at(i);
                        p2.x += x * roi_width + img_2_src.size[1];
                        p2.y += y * roi_height;

                        cv::Point2f p3 = keypoints_2_1.at(i);
                        p3.x += x * roi_width;
                        p3.y += y * roi_height;

                        if (is_display_all_line)
                        {
                            cv::line(img_out, p1, p2, cv::Scalar(0, 255, 255));
                        }

                        if (is_display_all_point) 
                        {
                            cv::circle(
                                img_out, p1, 5, cv::Scalar(255, 0, 0), -1);
                            cv::circle(
                                img_out, p3, 3, cv::Scalar(0, 0, 255), -1);
                        }
                    }
                }
            }
        }
    }

    // Visualize all list_key_points.
    for (size_t i = 0; i < prev_keypoints.size(); i++)
    {
        cv::Point pt;
        pt.x = next_keypoints[i].x + img_2_src.size[1];
        pt.y = next_keypoints[i].y;

        cv::line(
            img_out, prev_keypoints[i], pt, cv::Scalar(0, 255, 0));
    }

    std::cout << "Time Elapsed: " << time_used.count() << "\n";
    std::cout << "Number of Pair(s) Found: " << prev_keypoints.size() << "\n";
    cv::imwrite("out.png", img_out);
    cv::imshow("KLT Tracker", img_out);
    cv::waitKey(0);

    return 0;
}
