#include "frontend/VisualFrontend.h"
#include <chrono>
using namespace std;
using namespace cv;

void VisualFrontend::downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void VisualFrontend::downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

VisualFrontend::VisualFrontend()
{
	//Initialise detector
//	std::string detectorType = "Feature2D.BRISK";
//
//	detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", thresholdExtraction);
	//Initialize ID
    gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);

    d_pyrLK.winSize.width = 21;
    d_pyrLK.winSize.height = 21;
    d_pyrLK.maxLevel = 3;
    d_pyrLK.iters = 30;

	newId = 0;
}

void VisualFrontend::trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints)
{
	if (oldPoints.size() > 0)
	{
        //Track prevoius points with optical flow
		auto festart = chrono::steady_clock::now();
		track1(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;


		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	extract1(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

	//save old image
	im_prev = im_gray;

}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    vector<Point2f> newPointsVector;
	 GpuMat d_frame0Gray(im_gray);
    GpuMat d_prevPts;

    gpu_detector(d_frame0Gray, d_prevPts);

    downloadpts(d_prevPts, newPointsVector);
    
    grid.setImageSize1(im_gray.cols, im_gray.rows);

	//Prepare grid
#pragma omp parallel
	for (Point2f& oldPoint : oldPoints)
	{
		grid.addPoint1(oldPoint);
	}
	for (auto point : newPointsVector)
	{
		if (grid.isNewFeature(point))
		{
			oldPoints.addPoint(point, newId);
			newPoints.addPoint(point, newId);
			newId++;
		}
	}
	grid.resetGrid1();
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    vector<Point2f> points = oldPoints.getPoints();
    vector<float> fb_err;
    Mat prevPts = Mat(1,points.size(), CV_32FC2);
    

    for(size_t i=0;i<points.size();i++){
        prevPts.at<Vec2f>( 0, i )[0]=points[i].x;
        prevPts.at<Vec2f>( 0, i )[1]=points[i].y;
    }



    GpuMat d_frame0(im_prev);
    GpuMat d_frame1(im_gray);
    GpuMat d_nextPts;
    GpuMat d_status,d_status_back;
    GpuMat d_pts_back;


    GpuMat d_prevPts(prevPts);

    d_pyrLK.sparse(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);

    vector<Point2f> PrevPointsVec;
    downloadpts(d_prevPts, PrevPointsVec);
    
    vector<Point2f> nextPts(d_nextPts.cols);
    downloadpts(d_nextPts, nextPts);

    vector<uchar> status(d_status.cols);
    downloadmask(d_status, status);

    d_pyrLK.sparse(d_frame1, d_frame0, d_nextPts, d_pts_back, d_status_back);

    vector<Point2f> pts_back(d_pts_back.cols);
    downloadpts(d_pts_back, pts_back);

    vector<uchar> status_back(d_status_back.cols);
    downloadmask(d_status_back, status_back);
    
    for (size_t i = 0; i < status.size(); i++) 
    {
      float distance = std::sqrt(
                std::pow(PrevPointsVec[i].x - pts_back[i].x, 2) + 
                std::pow(PrevPointsVec[i].y - pts_back[i].y, 2));
        if (status.at(i) == 1 && status_back.at(i) == 1 && distance < thresholdFBError )
        {
         status[i] = 1;
        }
        else
        {
        status[i] = 0;
        }
    }

    trackedPoints = Features2D(oldPoints, nextPts, status);}
