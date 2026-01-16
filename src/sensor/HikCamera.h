#pragma once
#include "ImageSensor.h"
#include "PlogInitializer.h"

#define byte hik_byte_temp
#define ACCESS_MASK hik_access_mask_temp
#include "HCNetSDK.h"
#undef byte
#undef ACCESS_MASK

class HikCamera : public ImageSensor
{
public:
    HikCamera(const std::string &_device_ip, int _queue_max_length, bool _is_full_drop,
              int _capture_interval_ms, HWND _preview_handler);

private:
    virtual void dataCollectionLoop() override;
    void initDevice();
    void login();
    void releaseDevice();
    void startReplay();

    std::string device_ip;
    int device_port;
    std::string device_userName;
    std::string device_password;
    long userID;
    long replayHandler;
    HWND previewHandler;
};