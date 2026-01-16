#pragma once
#include "ImageSensor.h"

class LocalImage : public ImageSensor
{
public:
    LocalImage(std::string _dir_path, int _queue_max_length, bool _is_full_drop);
    virtual cv::Mat getData() override;
    virtual cv::Mat getDataNoBlock() override;

private:
    virtual void dataCollectionLoop() override;
    std::string dir_path;
};