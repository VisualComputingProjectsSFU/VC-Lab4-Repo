//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
void downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}
int main( int argc, char** argv )
{

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
	detector->set("thres", 100);


    detector->detect( img_1, kps );
    for ( auto kp:kps )
        keypoints.push_back( kp.pt );

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    vector<cv::Point2f> next_keypoints_klt;
    vector<cv::Point2f> prev_keypoints_klt;
    int g_x = 10;
    int g_y = 10;
    bool grid[g_x][g_y]= {false};
    cout<<"image size"<<img_1.size()<<endl;
    cout<<"image size"<<img_1.rows<<endl;
    cout<<"image size"<<img_1.cols<<endl;
    for ( auto kp:keypoints )
        {
            int norm_x = (int)round((kp.x/img_1.cols)*g_x)-1;
            int norm_y = (int)round((kp.y/img_1.rows)*g_y)-1;
           //  cout<<"normaal x"<<norm_x<<"y"<<norm_y<<endl;
           // cout<<kp.x<<"y"<<kp.y<<endl; 
           if (grid[norm_x][norm_y]==false)
           {
            prev_keypoints.push_back(kp);
            grid[norm_x][norm_y]=true;
            // cout<<"pushed"<<endl;
           }
           else
           {
            // cout<<"rejected"<<endl;
           }
        }
    vector<unsigned char> status;
    vector<float> error;

    //cpu
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time for forward："<<time_used.count()<<" seconds."<<endl;
    auto prev_keypoints_forward = prev_keypoints;

    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_2 , img_1, next_keypoints, prev_keypoints, status, error );
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time for backward："<<time_used.count()<<" seconds."<<endl;
    auto prev_keypoints_backward = prev_keypoints;

    for ( size_t i=0; i< prev_keypoints.size() ;i++)
    {
        //cout<<(int)status[i]<<endl;
        auto distance_fw_bw= sqrt(pow((prev_keypoints_backward[i].x - prev_keypoints_forward[i].x ),2)+pow((prev_keypoints_backward[i].y  - prev_keypoints_forward[i].y ),2));
        if(status[i] == 1 && distance_fw_bw<=0.2)
        {
            prev_keypoints_klt.push_back(prev_keypoints_forward[i]);
            next_keypoints_klt.push_back(next_keypoints[i]);    
        }
    }
    // visualize all  keypoints
    hconcat(img_1,img_2,img_1);
    for ( size_t i=0; i< prev_keypoints_klt.size() ;i++)
    {
        //cout<<(int)status[i]<<endl;
        for (int i=0 ;i<2*g_x;i++)
        {
            cv::Point2d p1=cv::Point2d(i*img_1.cols/(g_x*2),0);
            cv::Point2d p2=cv::Point2d(i*img_1.cols/(g_x*2),img_1.rows-1);
            line(img_1, p1,p2, cv::Scalar(255,255,255));
        }
        for (int i=0 ;i<g_y;i++)
        {
            cv::Point2d p1=cv::Point2d(0,i*img_1.rows/g_y);
            cv::Point2d p2=cv::Point2d(img_1.cols-1,i*img_1.rows/g_y);
            line(img_1, p1,p2, cv::Scalar(255,255,255));
        }
        

        
        Point pt;
        pt.x =  next_keypoints_klt[i].x + img_2.size[1];
        pt.y =  next_keypoints_klt[i].y;

        line(img_1, prev_keypoints_klt[i], pt, cv::Scalar(0,255,255));
        circle(img_1, prev_keypoints_klt[i], 5, cv::Scalar(0, 0, 255), -1);
        circle(img_1, pt, 5, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("klt tracker", img_1);
    //cv::waitKey(0);


    //gpu
    GpuMat gpu_img_1(img_1);
    GpuMat gpu_img_2(img_2);
    GpuMat gpu_prev_keypoints(prev_keypoints);
    GpuMat gpu_next_keypoints;
    GpuMat d_status,d_status_back;
    GpuMat gpu_points_back;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::PyrLKOpticalFlow.sparse( gpu_img_1, gpu_img_2, gpu_prev_keypoints, gpu_next_keypoints, d_status);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time for forward："<<time_used.count()<<" seconds."<<endl;

    t1 = chrono::steady_clock::now();
    cv::PyrLKOpticalFlow.sparse( gpu_img_2 , gpu_img_1, gpu_next_keypoints, gpu_points_back, d_status_back);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time for backward："<<time_used.count()<<" seconds."<<endl;
    
    vector<cv::Point2f> cpu_prev_keypoints;
    vector<cv::Point2f> cpu_next_keypoints;
    vector<cv::Point2f> cpu_points_back;
    vector<unsigned char> status_fwd;
    vector<unsigned char> status_bwd;

    downloadpts(gpu_prev_keypoints,cpu_prev_keypoints);
    downloadpts(gpu_next_keypoints,cpu_next_keypoints);
    downloadpts(gpu_points_back,cpu_points_back);

    downloadmask(d_status,status_fwd);
    downloadmask(d_status_back,status_bwd);

    for ( size_t i=0; i< prev_keypoints.size() ;i++)
    {
        //cout<<(int)status[i]<<endl;
        auto distance_fw_bw= sqrt(pow((cpu_points_back[i].x - cpu_prev_keypoints[i].x ),2)+pow((cpu_points_back[i].y  - cpu_prev_keypoints[i].y ),2));
        if(status_fwd[i] == 1 && status_bwd[i] == 1 && distance_fw_bw<=0.2)
        {
            prev_keypoints_klt.push_back(cpu_prev_keypoints[i]);
            next_keypoints_klt.push_back(cpu_next_keypoints[i]);    
        }
    }
    // visualize all  keypoints
    hconcat(img_1,img_2,img_1);
    for ( size_t i=0; i< prev_keypoints_klt.size() ;i++)
    {
        //cout<<(int)status[i]<<endl;
        for (int i=0 ;i<2*g_x;i++)
        {
            cv::Point2d p1=cv::Point2d(i*img_1.cols/(g_x*2),0);
            cv::Point2d p2=cv::Point2d(i*img_1.cols/(g_x*2),img_1.rows-1);
            line(img_1, p1,p2, cv::Scalar(255,255,255));
        }
        for (int i=0 ;i<g_y;i++)
        {
            cv::Point2d p1=cv::Point2d(0,i*img_1.rows/g_y);
            cv::Point2d p2=cv::Point2d(img_1.cols-1,i*img_1.rows/g_y);
            line(img_1, p1,p2, cv::Scalar(255,255,255));
        }
        

        Point pt;
        pt.x =  next_keypoints_klt[i].x + img_2.size[1];
        pt.y =  next_keypoints_klt[i].y;

        line(img_1, prev_keypoints_klt[i], pt, cv::Scalar(0,255,255));
        circle(img_1, prev_keypoints_klt[i], 5, cv::Scalar(0, 0, 255), -1);
        circle(img_1, pt, 5, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("klt tracker", img_1);
    cv::waitKey(0);

    // //before KLT
    // for ( size_t i=0; i< prev_keypoints_forward.size() ;i++)
    // {
    //     //cout<<(int)status[i]<<endl;
    //     if(status[i] == 1)
    //     {
    //         Point pt;
    //         pt.x =  next_keypoints[i].x + img_2.size[1];
    //         pt.y =  next_keypoints[i].y;

    //         line(img_1, prev_keypoints_forward[i], pt, cv::Scalar(0,255,255));
    //         circle(img_1, prev_keypoints_forward[i], 5, cv::Scalar(0, 0, 255), -1);
    //         circle(img_1, pt, 5, cv::Scalar(0, 255, 0), -1);
    //     }
    // }
    // // cv::imshow("klt tracker", img_1);
    // cv::waitKey(0);

    return 0;
}
