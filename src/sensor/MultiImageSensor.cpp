#include "MultiImageSensor.h"

MultiImageSensor::MultiImageSensor()
{
}

MultiImageSensor::~MultiImageSensor()
{
    this->stopAll();
}

void MultiImageSensor::addSensor(std::shared_ptr<ImageSensor> sensor)
{
    if (!sensor)
    {
        PLOGW << "Attempting to add null sensor";
        return;
    }

    std::unique_lock<std::mutex> lock(mutex);
    sensors.push_back(sensor);
    PLOGI << "Sensor added. Total sensors: " << sensors.size();
}

void MultiImageSensor::removeSensor(size_t index)
{
    std::unique_lock<std::mutex> lock(mutex);

    if (index >= sensors.size())
    {
        PLOGW << "Sensor index " << index << " out of range";
        return;
    }

    sensors[index]->stop();
    sensors.erase(sensors.begin() + index);
    PLOGI << "Sensor at index " << index << " removed. Total sensors: " << sensors.size();
}

void MultiImageSensor::startAll()
{
    std::unique_lock<std::mutex> lock(mutex);

    for (auto &sensor : sensors)
    {
        sensor->start();
    }

    PLOGI << "All " << sensors.size() << " sensors started";
}

void MultiImageSensor::stopAll()
{
    std::unique_lock<std::mutex> lock(mutex);

    for (auto &sensor : sensors)
    {
        sensor->stop();
    }

    PLOGI << "All sensors stopped";
}

void MultiImageSensor::clearAll()
{
    std::unique_lock<std::mutex> lock(mutex);

    for (auto &sensor : sensors)
    {
        sensor->clear();
    }

    PLOGV << "All sensors cleared";
}

std::vector<cv::Mat> MultiImageSensor::getAllData()
{
    std::vector<cv::Mat> result;

    std::unique_lock<std::mutex> lock(mutex);

    for (int i = 0; i < sensors.size(); ++i)
    {
        auto &sensor = sensors[i];

        if (!sensor->isRunning())
            continue;
        cv::Mat img = sensor->getLastestFrame();
        if (img.empty())
        {
            PLOGW << "No data available from sensor: " << i;
            continue;
        }
        result.push_back(img);
    }

    return result;
}

cv::Mat MultiImageSensor::getDataByIndex(size_t index)
{
    std::unique_lock<std::mutex> lock(mutex);

    if (index >= sensors.size())
    {
        PLOGW << "Sensor index " << index << " out of range, returning empty Mat";
        return cv::Mat();
    }

    return sensors[index]->getDataNoBlock();
}

cv::Mat MultiImageSensor::getLatestByIndex(size_t index)
{
    std::unique_lock<std::mutex> lock(mutex);

    if (index >= sensors.size())
    {
        PLOGW << "Sensor index " << index << " out of range, returning empty Mat";
        return cv::Mat();
    }

    return sensors[index]->getLastestFrame();
}

size_t MultiImageSensor::getSensorCount() const
{
    std::unique_lock<std::mutex> lock(mutex);
    return sensors.size();
}
