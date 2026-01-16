#pragma once
#include "ImageSensor.h"

/**
 * @brief 从本地视频文件中读取帧的传感器类
 * @details 继承自 ImageSensor，可以设置帧间隔参数来控制每隔几帧才往队列里加一帧
 */
class LocalVideo : public ImageSensor
{
public:
    /**
     * @brief 构造函数
     * @param _video_path 视频文件路径
     * @param _queue_max_length 队列最大长度
     * @param _is_full_drop 队列满时是否丢弃最旧的帧
     * @param _frame_skip 帧间隔，每隔 _frame_skip 帧才往队列加一帧（默认为1，表示每帧都加）
     */
    LocalVideo(std::string _video_path,
               int _queue_max_length,
               bool _is_full_drop,
               int _frame_skip);

    virtual ~LocalVideo();

    /**
     * @brief 设置帧间隔
     * @param frame_skip 每隔 frame_skip 帧才往队列加一帧
     */
    void setFrameSkip(int frame_skip);

private:
    virtual void dataCollectionLoop() override;

    std::string video_path;
    int frame_skip; // 帧间隔，1表示每帧都要，2表示每隔1帧加一次，以此类推
};
