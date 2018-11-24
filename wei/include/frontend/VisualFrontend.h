#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef INCLUDE_VISUALFRONTEND_H_
#define INCLUDE_VISUALFRONTEND_H_

#include "core/Features.h"
#include "OccupancyGrid.h"

class VisualFrontend
{
    public:
        VisualFrontend(size_t col, size_t row);

        void trackAndExtract(
            cv::Mat &im_gray, 
            Features2D &tracked_points, 
            Features2D &new_points);

        inline Features2D &getCurrentFeatures()
        {
            return old_points;
        }

        // Utils.
        static void downloadMasks(
            const GpuMat &d_mat, std::vector<uchar> &vec);
        static void downloadPoints(
            const GpuMat &d_mat, std::vector<cv::Point2f> &vec);

    protected:
        void extract(cv::Mat &im_gray, Features2D &new_points);
        void track(cv::Mat &im_gray, Features2D &points);

        // Implement two functions below.
        void extract1(cv::Mat &im_gray, Features2D &new_points);
        void track1(cv::Mat &im_gray, Features2D &points);

    protected:

        // Extracted data.
        Features2D old_points;
        cv::Mat im_prev;

    private:

	// Configurations.
	const bool is_gpu = true;

        // Unique ID for each point.
        unsigned int new_id;

        // KLT tracker and feature detector.
        cv::Ptr<cv::FeatureDetector> detector_cpu;
        GoodFeaturesToTrackDetector_GPU detector_gpu;
        PyrLKOpticalFlow tracker_gpu;

        // Data needed to guide extraction.
        OccupancyGrid grid;

        // Parameters.
        const int threshold_extraction = 20;
        const double threshold_error = 1.0;
        const double threshold_validation = 0.05;
};

#endif /* INCLUDE_VISUALFRONTEND_H_ */
