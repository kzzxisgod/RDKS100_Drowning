#include "LocalVideo.h"
#include "Config_FileManager.h"

LocalVideo::LocalVideo(std::string _video_path,
                       int _queue_max_length,
                       bool _is_full_drop,
                       int _frame_skip)
    : ImageSensor(_queue_max_length, 0, _is_full_drop),
      video_path(_video_path),
      frame_skip(_frame_skip > 1 ? _frame_skip : 1)
{
    if (frame_skip < 1)
    {
        PLOGW << "frame_skip must be >= 1, setting to 1";
        this->frame_skip = 1;
    }
    this->capture_interval_ms = Config_FileManager::get_instance()->runtimeConfig.video_capture_interval;
    PLOGI << "Video capture interval set to " << this->capture_interval_ms << "ms";
}

LocalVideo::~LocalVideo()
{
    this->stop();
}

void LocalVideo::setFrameSkip(int new_frame_skip)
{
    if (new_frame_skip < 1)
    {
        PLOGW << "frame_skip must be >= 1, ignoring";
        return;
    }
    frame_skip = new_frame_skip;
    PLOGV << "Frame skip set to " << frame_skip;
}

void LocalVideo::dataCollectionLoop()
{
    // 打开视频文件
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened())
    {
        PLOGE << "Failed to open video file: " << video_path;
        return;
    }

    PLOGV << "Video file opened: " << video_path;

    // 获取视频信息
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    PLOGI << "Video info: Total frames: " << total_frames
          << ", FPS: " << fps
          << ", Resolution: " << frame_width << "x" << frame_height;

    int skip_count = 0;

    // 循环读取视频帧
    while (this->is_running)
    {
        bool ret = cap.grab();
        if (!ret)
        {
            // 到达视频末尾，重新开始或退出
            PLOGV << "End of video reached";
            // 重置视频到开头
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            skip_count = 0;
            continue;
        }

        skip_count++;
        if (skip_count == frame_skip)
        {
            skip_count = 0;
            cv::Mat frame;
            cap.retrieve(frame);
            this->enqueueData(frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(capture_interval_ms));
        }
    }

    cap.release();
    PLOGV << "Video capture stopped";
}
