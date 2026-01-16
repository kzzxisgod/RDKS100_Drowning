#include "Config_FileManager.h"
#include <fstream>

Config_FileManager *Config_FileManager::instance = nullptr;
std::mutex Config_FileManager::instance_mutex;

Config_FileManager* Config_FileManager::get_instance()
{
    if (instance == nullptr)
    {
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (instance == nullptr)
        {
            instance = new Config_FileManager();
        }
    }
    return instance;
}

void Config_FileManager::write_file()
{
    std::ofstream os(file_path);
    cereal::JSONOutputArchive archive(os);

    archive(cereal::make_nvp("DirConfig", this->dirConfig));
    archive(cereal::make_nvp("RuntimeConfig", this->runtimeConfig));
}

void Config_FileManager::read_file()
{   
    std::ifstream is(file_path);
    cereal::JSONInputArchive archive(is);

    archive(cereal::make_nvp("DirConfig", this->dirConfig));
    archive(cereal::make_nvp("RuntimeConfig", this->runtimeConfig));

}
