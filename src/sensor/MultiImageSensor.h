#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "opencv2/opencv.hpp"
#include "ImageSensor.h"

/**
 * @brief MultiImageSensor 是一个多图像传感器容器类
 * 
 * 用于管理多个 ImageSensor 对象，能够同时从多个传感器获取图像数据，
 * 用于图像拼接等应用场景
 */
class MultiImageSensor
{
public:
    /**
     * @brief 构造函数
     */
    MultiImageSensor();
    
    /**
     * @brief 析构函数
     */
    ~MultiImageSensor();
    
    /**
     * @brief 添加一个图像传感器
     * 
     * @param sensor 指向 ImageSensor 的智能指针
     */
    void addSensor(std::shared_ptr<ImageSensor> sensor);
    
    /**
     * @brief 移除指定位置的传感器
     * 
     * @param index 传感器在容器中的索引
     */
    void removeSensor(size_t index);
    
    /**
     * @brief 启动所有传感器
     */
    void startAll();
    
    /**
     * @brief 停止所有传感器
     */
    void stopAll();
    
    /**
     * @brief 清空所有传感器的缓冲队列
     */
    void clearAll();
    
    /**
     * @brief 同时从所有传感器获取一张图像
     * 
     * @return 返回 vector，包含所有传感器的图像
     *         如果某个传感器没有可用图像，对应位置为空的 Mat
     */
    std::vector<cv::Mat> getAllData();
    
    /**
     * @brief 获取指定索引传感器的图像
     * 
     * @param index 传感器索引
     * @return 返回该传感器的一张图像；如果索引无效或没有可用图像，返回空的 Mat
     */
    cv::Mat getDataByIndex(size_t index);

    cv::Mat getLatestByIndex(size_t index);
    
    /**
     * @brief 获取传感器数量
     * 
     * @return 当前容器中的传感器数量
     */
    size_t getSensorCount() const;

    std::vector<std::shared_ptr<ImageSensor>> sensors;

private:
    mutable std::mutex mutex;
};
