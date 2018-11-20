#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef OCCUPANCYGRID_H_
#define OCCUPANCYGRID_H_

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

class OccupancyGrid
{
    public:
        OccupancyGrid();

        void initializer();
        
        // Compute the size of cell (ix and iy) according to given image size 
        // and grid resolution.
        void setImageSize(size_t col, size_t row);

        // Update the OccupancyGrid (isFree) when adding new features.
        void addPoint(cv::Point2f& p); 
        
        // P is a newfeature when cell / neighbour cells are all free.
        bool isNewFeature(cv::Point2f& p); 

        // Rest isFree.
        void resetGrid(); 

        // Print current occupency grid.
        friend std::ostream &operator<<(
            std::ostream &os, const OccupancyGrid &grid); 

        // Implementing these 5 functions below.
        void initialize1();
        void setImageSize1(size_t col, size_t row);
        void addPoint1(cv::Point2f& p);
        bool isNewFeature1(cv::Point2f& p);
        void resetGrid1();

    private:

        // Number of cells.
        static const size_t nx = 32;
        static const size_t ny = 20;

        // Data needed by the algorithm.
        bool isFree[ny][nx];
        size_t ix;
        size_t iy;

        // Cell sizes.
        double cx;
        double cy;

        // Mask to define nearby features.
        const std::array<cv::Point, 9>  nearby {
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1), 
            cv::Point(-1, 0),  cv::Point(0, 0),  cv::Point(1, 0), 
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1)
        };
};

// Print current occupency grid.
inline std::ostream &operator<<(std::ostream &os, const OccupancyGrid &grid)  
{  
    for (size_t y = 0; y < grid.ny; y++)
    {
        for (size_t x = 0; x < grid.nx; x++)
        {
            if (grid.isFree[y][x])
                os << "T ";
            else
                os << "F ";
        }
        os << "\n";
    }
    return os;  
}

#endif /* OCCUPANCYGRID_H_ */
