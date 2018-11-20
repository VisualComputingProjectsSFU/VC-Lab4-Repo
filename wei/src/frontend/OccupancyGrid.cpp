#include "frontend/OccupancyGrid.h"
#include <iostream>

OccupancyGrid::OccupancyGrid()
{
    initialize1();
}

void OccupancyGrid::initialize1()
{
    resetGrid1();
}

void OccupancyGrid::setImageSize1(size_t col, size_t row)
{
    ix = col;
    iy = row;
    cx = ix * 1.0 / nx;
    cy = iy * 1.0 / ny;
}

void OccupancyGrid::addPoint1(cv::Point2f &p)
{
    for (size_t i = 0; i < nearby.size(); i++)
    {
        int x = (int)(p.x / cx) + nearby[i].x;
        int y = (int)(p.y / cy) + nearby[i].y;
        if (x >= 0 && x < (int)nx && y >= 0 && y < (int)ny)
        {
            isFree[y][x] = false;
        }
    }
}

bool OccupancyGrid::isNewFeature1(cv::Point2f &p)
{
    for (size_t i = 0; i < nearby.size(); i++)
    {
        int x = (int)(p.x / cx) + nearby[i].x;
        int y = (int)(p.y / cy) + nearby[i].y;
        if ((x >= 0 && x < (int)nx && y >= 0 && y < (int)ny) && 
            isFree[y][x] == false)
        {
            return false;
        }
    }
    return true;
}

void OccupancyGrid::resetGrid1()
{
    for (size_t y = 0; y < ny; y++)
    {
        for (size_t x = 0; x < nx; x++)
        {
            isFree[y][x] = true;
        }
    }
}
