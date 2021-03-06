#include "frontend/VisualFrontend.h"
#include <chrono>

void VisualFrontend::downloadPoints(
    const GpuMat &d_mat, std::vector<cv::Point2f> &vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*) &vec[0]);
    d_mat.download(mat);
}

void VisualFrontend::downloadMasks(
    const GpuMat &d_mat, std::vector<uchar> &vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*) &vec[0]);
    d_mat.download(mat);
}

VisualFrontend::VisualFrontend(size_t col, size_t row)
{
    // Initialize grid.
    grid.initialize1();
    grid.setImageSize1(col, row);

    // Initialise detector.
    std::string detectorType = "Feature2D.BRISK";
    detector_cpu = cv::Algorithm::create<cv::FeatureDetector>(detectorType);
    detector_cpu->set("thres", threshold_extraction);
    detector_gpu = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);

    // Initialize tracker.
    tracker_gpu.winSize.width = 21;
    tracker_gpu.winSize.height = 21;
    tracker_gpu.maxLevel = 3;
    tracker_gpu.iters = 30;

    new_id = 0;
}

void VisualFrontend::trackAndExtract(
    cv::Mat &im_gray, Features2D &tracked_points, Features2D &new_points)
{
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    if (old_points.size() > 0)
    {
        // Track prevoius points with optical flow.
        start = std::chrono::steady_clock::now();
        track1(im_gray, tracked_points);
        end = std::chrono::steady_clock::now();
        std::cout << "KLT Running Time: ";
        std::cout << std::chrono::duration<double, std::milli>(
            end - start).count();
        std::cout << " ms" << "\n";

        // Save tracked points.
        old_points = tracked_points;
    }

    // Extract new points.
    start = std::chrono::steady_clock::now();
    extract1(im_gray, new_points);
    end = std::chrono::steady_clock::now();
    std::cout << "New feature time: ";
    std::cout << std::chrono::duration<double, std::milli>(
        end - start).count();
    std::cout << " ms" << "\n";

    // Save old image.
    im_prev = im_gray;
}

void VisualFrontend::extract1(cv::Mat &im_gray, Features2D &new_points)
{
    // Detect features.
    std::vector<cv::Point2f> vector_points;
    
    if (is_gpu)
    {
    	GpuMat d_vector_points;
	    GpuMat d_im_gray(im_gray);
	    detector_gpu(d_im_gray, d_vector_points);
	    downloadPoints(d_vector_points, vector_points);
    }
    else
    {
        std::vector<cv::KeyPoint> vector_key_points;
        detector_cpu->detect(im_gray, vector_key_points);
        for (auto kp : vector_key_points)
        {
            vector_points.push_back(kp.pt);
        }
    }

    for (auto p : vector_points)
    {
        // Update new point if grid is free.
        if (grid.isNewFeature1(p))
        {
            new_points.addPoint(p, new_id);
            old_points.addPoint(p, new_id);
            new_id++;
        }
    }

    // Reset the grid.
    grid.resetGrid1();
}

void VisualFrontend::track1(cv::Mat &im_gray, Features2D &tracked_points)
{
    // Initialize keypoints.
    std::vector<float> error_1;
    std::vector<float> error_2;
    std::vector<unsigned char> status_1;
    std::vector<unsigned char> status_2;
    std::vector<cv::Point2f> keypoints_1;
    std::vector<cv::Point2f> keypoints_1_2;
    std::vector<cv::Point2f> keypoints_2_1;
    for (size_t i = 0; i < old_points.size(); i++)
    {
        keypoints_1.push_back(old_points[i]);
    }

    if (is_gpu)
    {
        // Cross validation.
        cv::Mat mat_p = cv::Mat(1, keypoints_1.size(), CV_32FC2);
        for (size_t i = 0; i < keypoints_1.size(); i++)
        {
            mat_p.at<cv::Vec2f>(0, i)[0] = keypoints_1[i].x;
            mat_p.at<cv::Vec2f>(0, i)[1] = keypoints_1[i].y;
        }
        GpuMat d_im_prev(im_prev);
        GpuMat d_im_gray(im_gray);
        GpuMat d_keypoints_1(mat_p);
        GpuMat d_keypoints_1_2;
        GpuMat d_keypoints_2_1;         
        GpuMat d_status_1;
        GpuMat d_status_2;
        
        tracker_gpu.sparse(
            d_im_prev, d_im_gray, d_keypoints_1, d_keypoints_1_2, d_status_1);
        tracker_gpu.sparse(
            d_im_gray, d_im_prev, d_keypoints_1_2, d_keypoints_2_1, d_status_2);

        VisualFrontend::downloadPoints(d_keypoints_1, keypoints_1);
        VisualFrontend::downloadPoints(d_keypoints_1_2, keypoints_1_2);
        VisualFrontend::downloadPoints(d_keypoints_2_1, keypoints_2_1);
        VisualFrontend::downloadMasks(d_status_1, status_1);
        VisualFrontend::downloadMasks(d_status_2, status_2);
    }
    else
    {
        // Cross validation.
        cv::calcOpticalFlowPyrLK(
            im_prev, im_gray, keypoints_1, keypoints_1_2, status_1, error_1);
        cv::calcOpticalFlowPyrLK(
            im_gray, im_prev, keypoints_1_2, keypoints_2_1, status_2, error_2);
    }

    for (size_t i = 0; i < status_1.size(); i++) 
    {
        if (status_1.at(i) == 1 && status_2.at(i) == 1)
        {
            float distance = std::sqrt(
                std::pow(keypoints_1[i].x - keypoints_2_1[i].x, 2) + 
                std::pow(keypoints_1[i].y - keypoints_2_1[i].y, 2));
            if (distance < threshold_validation)
            {
                cv::Point2f p = keypoints_1_2.at(i);
                tracked_points.addPoint(p, new_id++);
                grid.addPoint1(p);
            }
        }
    }
}
