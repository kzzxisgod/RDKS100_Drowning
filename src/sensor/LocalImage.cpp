#include "LocalImage.h"
#include <filesystem>

LocalImage::LocalImage(std::string _dir_path, int _queue_max_length, bool _is_full_drop)
    : ImageSensor(_queue_max_length, 500, _is_full_drop),
      dir_path(_dir_path)
{
}

// 从队列头部读取出一帧数据
cv::Mat LocalImage::getData()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (this->images.empty())
    {
        PLOGV << "Blocking wait for new data...";
        cv.wait(lock);
    }
    cv::Mat img = this->images.front();
    this->images.pop_front();
    return img;
}

cv::Mat LocalImage::getDataNoBlock()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (this->images.empty())
    {
        PLOGV << "No data available, returning empty Mat.";
        return cv::Mat();
    }
    cv::Mat img = this->images.front();
    this->images.pop_front();
    return img;
}


static bool ends_with(const std::string &str, const std::string &suffix)
{
    if (str.size() < suffix.size())
        return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void LocalImage::dataCollectionLoop()
{
    // 检查路径是否有效
    if (!std::filesystem::exists(dir_path))
    {
        PLOGE << "Directory does not exist: " << dir_path;
        return;
    }

    // 循环遍历dir_path目录下的所有文件
    while (this->is_running)
    {
        for (const auto &entry : std::filesystem::directory_iterator(dir_path))
        {
            if (entry.is_regular_file())
            {
                std::string file_path = entry.path().string();

                // 检查文件扩展名是否是图像文件
                if (ends_with(file_path, ".jpg") ||
                    ends_with(file_path, ".png") ||
                    ends_with(file_path, ".bmp"))
                {
                    cv::Mat img = cv::imread(file_path);
                    this->enqueueData(img);
                }
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(this->capture_interval_ms));

            if (!this->is_running)
                break;
        }
    }
}