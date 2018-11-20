#include "frontend/OccupancyGrid.h"
#include "frontend/VisualFrontend.h"
#include "core/Features.h"

int main(int argc, char *argv[])
{
	cv::Mat img_1_src = cv::imread("1.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2_src = cv::imread("2.png", CV_LOAD_IMAGE_COLOR);

    // Test of occupency grid.
    OccupancyGrid grid;

    grid.setImageSize1(640, 400);
    cv::Point2f p;
    p.x = 639;
    p.y = 399;
    grid.addPoint1(p);
    p.x = 640;
    p.y = 400;
    std::cout << grid.isNewFeature1(p) << "\n";
    std::cout << grid << "\n";

    // test of visual frontend.
    VisualFrontend frontend(img_1_src.cols, img_1_src.rows);
    Features2D old_points;
    Features2D new_points;

	std::cout << "FIRST" << "\n";
    frontend.trackAndExtract(img_1_src, old_points, new_points);
    
    std::cout << "OLD POINTS" << "\n";
    std::cout << old_points.getPoints().size() << "\n";
    for (auto p : old_points.getPoints())
    {
    	// std::cout << p << "\n";
    }
    std::cout << "NEW POINTS" << "\n";
    std::cout << new_points.getPoints().size() << "\n";
    for (auto p : new_points.getPoints())
    {
    	// std::cout << p << "\n";
    }

    std::cout << "SECOND" << "\n";
    frontend.trackAndExtract(img_2_src, old_points, new_points);
    
    std::cout << "OLD POINTS" << "\n";
    std::cout << old_points.getPoints().size() << "\n";
    for (auto p : old_points.getPoints())
    {
    	// std::cout << p << "\n";
    }
    std::cout << "NEW POINTS" << "\n";
    std::cout << new_points.getPoints().size() << "\n";
    for (auto p : new_points.getPoints())
    {
    	// std::cout << p << "\n";
    }
}
