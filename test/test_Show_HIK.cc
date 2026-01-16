#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib> // 用于 setenv

int main() {
    // 1. 强制 FFMPEG 使用 TCP 传输 (解决花屏和 Undefined type 报错)
    // 这必须在 VideoCapture 实例化之前设置
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);

    // 2. 使用子码流 (sub) 降低分辨率，减轻 X11 转发负担
    // 注意：将 main 改为 sub
    std::string rtsp_url = "rtsp://admin:waterline123456@192.168.127.15:554/h264/ch1/sub/av_stream";

    cv::VideoCapture cap;
    
    // 打开视频流，明确指定使用 FFMPEG 后端
    if (!cap.open(rtsp_url, cv::CAP_FFMPEG)) {
        std::cerr << "无法连接到摄像头，请检查 URL 或网络！" << std::endl;
        return -1;
    }

    std::cout << "TCP 连接成功，正在通过 X11 转发画面..." << std::endl;

    cv::Mat frame;
    cv::namedWindow("HIK_MobaXterm_Test", cv::WINDOW_AUTOSIZE);

    while (true) {
        if (!cap.read(frame)) {
            std::cout << "读取帧失败，正在重试..." << std::endl;
            continue;
        }

        // 3. 进一步缩小画面
        // 即使是子码流，通过 X11 转发也可能卡顿，缩小到 480p 以下会顺滑很多
        cv::Mat res_frame;
        cv::resize(frame, res_frame, cv::Size(480, 270)); 

        // 4. 显示画面
        cv::imshow("HIK_MobaXterm_Test", res_frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}