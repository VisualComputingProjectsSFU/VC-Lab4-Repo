#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

cv::Matx33d Findfundamental(
    std::vector<cv::Point2f> prev_subset, std::vector<cv::Point2f> next_subset)
{
    cv::Matx33d F;

    // Fill the blank.
    return F;
}

bool checkinlier(
    cv::Point2f prev_keypoint, 
    cv::Point2f next_keypoint, 
    cv::Matx33d Fcandidate, 
    double d)
{
    // Fill the blank.
    return false;
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    if (argc != 3)
    {
        std::cout << "Usage: feature_extraction img1 img2" << std::endl;
        return 1;
    }

    // Read two images.
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    std::list<cv::Point2f> keypoints;
    std::vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    cv::Ptr<cv::FeatureDetector>detector = 
        cv::Algorithm::create<cv::FeatureDetector>(detectorType);
    detector->set("thres", 100);

    detector->detect(img_1, kps);
    for (auto kp : kps)
        keypoints.push_back(kp.pt);

    std::vector<cv::Point2f> next_keypoints;
    std::vector<cv::Point2f> prev_keypoints;
    for (auto kp : keypoints)
        prev_keypoints.push_back(kp);
    std::vector<unsigned char> status;
    std::vector<float> error;
    std::chrono::steady_clock::time_point t1;
    t1 = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(
        img_1, img_2, prev_keypoints, next_keypoints, status, error);
    std::chrono::steady_clock::time_point t2;
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used;
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "LK Flow Time：" << time_used.count() << " seconds." << "\n";

    std::vector<cv::Point2f> kps_prev, kps_next;
    kps_prev.clear();
    kps_next.clear();
    for (size_t i = 0; i < prev_keypoints.size(); i++)
    {
        if (status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }

    // p Probability that at least one valid set of inliers is chosen.
    // d Tolerated distance from the model for inliers.
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(
        std::ceil(std::log(1.0 - p) / std::log(1.0 - std::pow(1.0 - e, 8))));
    cv::Mat Fundamental;
    cv::Matx33d F, Fcandidate;
    int bestinliers = -1;
    std::vector<cv::Point2f> prev_subset, next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

    for (int i = 0; i < niter; i++)
    {
        // Step 1: Randomly sample 8 matches for 8pt algorithm.
        std::unordered_set<int> rand_util;
        while (rand_util.size() < 8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        std::vector<int> random_indices(rand_util.begin(),rand_util.end());
        for (size_t j = 0; j < rand_util.size(); j++)
        {
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }

        // Step 2: Perform 8pt algorithm, get candidate F.
        Fcandidate = Findfundamental(prev_subset,next_subset);

        // Step 3: Evaluate inliers, decide if we need to update the solution.
        int inliers = 0;
        for (size_t j = 0; j < kps_prev.size(); j++)
        {
            if (checkinlier(
                prev_keypoints[j], next_keypoints[j], Fcandidate, d))
                inliers++;
        }
        if (inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // Step 4: After we finish all the iterations, use the inliers of the best
    // model to compute Fundamental matrix again.

    for (size_t j = 0; j < prev_keypoints.size(); j++)
    {
        if (checkinlier(kps_prev[j], kps_next[j], F, d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }

    }
    F = Findfundamental(prev_subset, next_subset);

    std::cout << "Fundamental Matrix: \n" << F << std::endl;
    return 0;
}
