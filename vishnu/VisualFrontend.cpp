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
		track(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;


		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	extract(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

	//save old image
	im_prev = im_gray;

}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    vector<Point2f> newPointsVector;
	//detector->detect(im_gray, newPointsVector);


    //GoodFeaturesToTrackDetector_GPU detector(300, 0.01, 0);
    GpuMat d_frame0Gray(im_gray);
    GpuMat d_prevPts;

    gpu_detector(d_frame0Gray, d_prevPts);

    downloadpts(d_prevPts, newPointsVector);

	//Prepare grid
	grid.setImageSize(im_gray.cols, im_gray.rows);

#pragma omp parallel
	for (Point2f& oldPoint : oldPoints)
	{
		grid.addPoint(oldPoint);
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
	grid.resetGrid();
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{//vector<unsigned char> status;
    //vector<unsigned char> status_back;
    //vector<Point2f> pts_back;
    //vector<Point2f> nextPts;
    //	vector<float> err;
    //	vector<float> err_back;

    vector<Point2f> points = oldPoints.getPoints();
    vector<float> fb_err;
    Mat prevPts = Mat(1,points.size(), CV_32FC2);
    //Calculate forward optical flow for prev_location


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

    vector<Point2f> nextPts(d_nextPts.cols);
    downloadpts(d_nextPts, nextPts);

    vector<uchar> status(d_status.cols);
    downloadmask(d_status, status);

    d_pyrLK.sparse(d_frame1, d_frame0, d_nextPts, d_pts_back, d_status_back);

    vector<Point2f> pts_back(d_pts_back.cols);
    downloadpts(d_pts_back, pts_back);

    vector<uchar> status_back(d_status_back.cols);
    downloadmask(d_status_back, status_back);


    //cpu version

//	calcOpticalFlowPyrLK(im_prev, im_gray, points, nextPts, status, err);
//	//Calculate backward optical flow for prev_location
//	calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, pts_back, status_back,
//				err_back);



    //Calculate forward-backward error
    for (size_t i = 0; i < points.size(); i++)
    {
        fb_err.push_back(norm(pts_back[i] - points[i]));
    }

    //Set status depending on fb_err and lk error
#pragma omp parallel
    for (size_t i = 0; i < status.size(); i++)
        status[i] = (fb_err[i] <= thresholdFBError) && status[i];

    trackedPoints = Features2D(oldPoints, nextPts, status);}
