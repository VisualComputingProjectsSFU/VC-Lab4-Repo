#include "frontend/OccupancyGrid.h"
using namespace std;
using namespace cv;

OccupancyGrid::OccupancyGrid()
{
	//initializer();
	initializer1();
}

void OccupancyGrid::initializer1()
{
    Iy = 1;
    Ix = 1;

    resetGrid1();
}

void OccupancyGrid::setImageSize1(size_t cols, size_t rows)
{
    Ix = cols / nx;
    Iy = rows / ny;
}

void OccupancyGrid::addPoint1(Point2f& p)
{
    size_t i = p.x / Ix;
    size_t j = p.y / Iy;

    if(i >= nx || j >= ny)
        return;

    isFree[i][j] = false;
}

bool OccupancyGrid::isNewFeature1(Point2f& p)
{
    int i = p.x / Ix;
    int j = p.y / Iy;

    unsigned int minX = std::max(0, i - 1);
    unsigned int maxX = std::min((int)nx, i + 2);

    unsigned int minY = std::max(0, j - 1);
    unsigned int maxY = std::min((int)ny, j + 2);

    for(unsigned int x = minX; x < maxX; x++)
     {
        for(unsigned int y = minY; y < maxY; y++)
        {
            if (isFree[x][y] == false)
            {
            return false;
            }    
        }
     }


    return true;
}

void OccupancyGrid::resetGrid1()
{
    for (size_t i = 0; i < nx; i++)
        for (size_t j = 0; j < ny; j++)
            isFree[i][j] = true;

}

