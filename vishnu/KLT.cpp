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
    bool grid[10][10]= {false};
    cout<<"image size"<<img_1.rows<<endl;
    cout<<"image size"<<img_1.cols<<endl;
    for ( auto kp:keypoints )
        {
            int norm_x = (int)round((kp.x/img_1.cols)*10)-1;
            int norm_y = (int)round((kp.y/img_1.rows)*10)-1;
            cout<<"normaal x"<<norm_x<<"y"<<norm_y<<endl;
           cout<<kp.x<<"y"<<kp.y<<endl; 
           if (grid[norm_x][norm_y]==false)
           {
            prev_keypoints.push_back(kp);
            grid[norm_x][norm_y]=true;
            cout<<"pushed"<<endl;
           }
           else
           {
            cout<<"rejected"<<endl;
           }
        }
    vector<unsigned char> status;
    vector<float> error;
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
        if(status[i] == 1 && distance_fw_bw<=0.02)
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
        if(status[i] == 1)
        {
            Point pt;
            pt.x =  next_keypoints_klt[i].x + img_2.size[1];
            pt.y =  next_keypoints_klt[i].y;

            line(img_1, prev_keypoints_klt[i], pt, cv::Scalar(0,255,255));
        }
    }

    cv::imshow("klt tracker", img_1);
    cv::waitKey(0);

    return 0;
}
