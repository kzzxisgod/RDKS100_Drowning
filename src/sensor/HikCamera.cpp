#include "HikCamera.h"
#include <exception>

HikCamera::HikCamera(const std::string &_device_ip, int _queue_max_length, bool _is_full_drop,
                     int _capture_interval_ms, HWND _preview_handler)
    : ImageSensor(_queue_max_length, _capture_interval_ms, _is_full_drop),
      previewHandler(_preview_handler),
      device_ip(_device_ip),
      device_port(8000),
      device_userName("admin"),
      device_password("waterline123456")
{
    assert(previewHandler != nullptr);
}

void HikCamera::dataCollectionLoop()
{
    try
    {
        this->initDevice();
        this->login();
        this->startReplay();
    }
    catch (std::exception &e)
    {
        PLOG_ERROR << e.what();
        this->is_running = false;
        this->releaseDevice();
        return;
    }

    // 延时启动,等待初始化完毕再采图
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    while (this->is_running)
    {
        constexpr DWORD img_buffer_size = 2560 * 2000 * 3; // 足够大就行了
        std::vector<char> img_buffer(img_buffer_size);
        DWORD offset = 0;
        bool ret = NET_DVR_CapturePictureBlock_New(this->replayHandler, img_buffer.data(),
                                                   img_buffer_size, &offset);
        if (ret == false)
        {
            PLOG_ERROR << "NET_DVR_CapturePictureBlock_New error, error code: " << NET_DVR_GetLastError();
            continue;
        }

        std::vector<char> raw_data(img_buffer.data(), img_buffer.data() + offset);
        cv::Mat img = cv::imdecode(cv::Mat(raw_data), cv::IMREAD_COLOR);
        resize(img, img, cv::Size(1280, 720));
        this->enqueueData(img);

        std::this_thread::sleep_for(std::chrono::milliseconds(this->capture_interval_ms));
    }

    this->releaseDevice();
}

void HikCamera::initDevice()
{
    bool ret = true;
    // 初始化
    ret = NET_DVR_Init();
    // 设置连接时间与重连时间
    ret = NET_DVR_SetConnectTime(2000, 1);
    ret = NET_DVR_SetReconnect(10000, true);

    if (ret == false)
    {
        PLOG_ERROR << "initDevice error, error code: " << NET_DVR_GetLastError();
        throw std::runtime_error("initDevice error");
    }
}

void HikCamera::login()
{
    // 登录参数，包括设备地址、登录用户、密码等
    NET_DVR_USER_LOGIN_INFO loginInfo = {0};
    loginInfo.bUseAsynLogin = 0;                                // 同步登录方式
    strcpy(loginInfo.sDeviceAddress, this->device_ip.c_str());  // 设备IP地址
    loginInfo.wPort = 8000;                                     // 设备服务端口
    strcpy(loginInfo.sUserName, this->device_userName.c_str()); // 设备登录用户名
    strcpy(loginInfo.sPassword, this->device_password.c_str()); // 设备登录密码

    // 设备信息, 输出参数
    NET_DVR_DEVICEINFO_V40 deviceInfoV40 = {0};

    // 登录设备
    this->userID = NET_DVR_Login_V40(&loginInfo, &deviceInfoV40);
    if (this->userID < 0)
    {
        PLOG_ERROR << "Login failed, error code:" << NET_DVR_GetLastError();
        throw std::runtime_error("login error");
    }
}

void HikCamera::startReplay()
{
    // 码流相关参数设置
    NET_DVR_PREVIEWINFO playInfo = {0};
    playInfo.hPlayWnd = this->previewHandler; // 需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
    playInfo.lChannel = 1;                    // 预览通道号
    playInfo.dwStreamType = 0;                // 0-主码流，1-子码流，2-码流3，3-码流4，以此类推
    playInfo.dwLinkMode = 0;                  // 0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP
    playInfo.bBlocked = 1;                    // 0- 非阻塞取流，1- 阻塞取流

    // 启动预览
    this->replayHandler = NET_DVR_RealPlay_V40(this->userID, &playInfo, NULL, NULL);
    if (this->replayHandler < 0)
    {
        PLOG_ERROR << "startReplay failed, error code:" << NET_DVR_GetLastError();
        throw std::runtime_error("startReplay error");
    }
}

void HikCamera::releaseDevice()
{
    // 关闭预览
    NET_DVR_StopRealPlay(this->replayHandler);
    // 注销用户
    NET_DVR_Logout(this->userID);
    // 释放SDK资源
    NET_DVR_Cleanup();
}