#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef INCLUDE_VISUALFRONTEND_H_
#define INCLUDE_VISUALFRONTEND_H_

#include "core/Features.h"
#include "OccupancyGrid.h"
using namespace std;
using namespace cv;

class VisualFrontend
{
public:
	VisualFrontend();

	void trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints);

	inline Features2D& getCurrentFeatures()
	{
		return oldPoints;
	}

	// utils
	static void downloadmask(const GpuMat& d_mat, vector<uchar>& vec);
	static void downloadpts(const GpuMat& d_mat, vector<Point2f>& vec);


protected:
	void extract(cv::Mat& im_gray, Features2D& newPoints);
	void track(cv::Mat& im_gray, Features2D& points);

	// Implement two functions below
	void extract1(Mat& im_gray, Features2D& newPoints);
	void track1(cv::Mat& im_gray, Features2D& points);

protected:
	//extracted data
	Features2D oldPoints;
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
