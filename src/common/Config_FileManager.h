#pragma once
#include "Config.h"
#include <mutex>

class Config_FileManager
{
public:
    static Config_FileManager *get_instance();
    void write_file();
    void read_file();

    DirConfig dirConfig;
    RuntimeConfig runtimeConfig;

private:
    Config_FileManager() {}
    static Config_FileManager* instance;
    static std::mutex instance_mutex;
    std::string file_path = "F:/MasterGraduate/03-Code/PanoramicTracking/configs/global_config.json";
};