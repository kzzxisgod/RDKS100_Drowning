#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "ultralytics_yolo11.hpp" // yolo目标检测
#include "ThreadPool.h" // 线程池
#include <ThreadSafeQueue.h> // 线程安全队列

class VideoWorker : public QObject {
    Q_OBJECT
public:
    VideoWorker(const std::string& url, const std::string& model, const std::string& labels);
    ~VideoWorker() = default;

    void process();
    void stop();

signals:
    // frameReadyLeft: 发送原始画面
    // frameReadyRight: 发送处理后(带框)的画面
    void frameReadyLeft(QImage img);
    void frameReadyRight(QImage img);

private:
    // 生产者逻辑：拉流与解码
    void captureLoop();
    // 消费者逻辑：退了和绘图
    void inferenceLoop();

    std::string rtsp_url;
    std::string model_path;
    std::string label_file;
    std::atomic<bool> is_running{false};

    // 核心缓冲区：限制大小为1或2，确保实时性
    ThreadSafeQueue<cv::Mat> frame_queue{1};

};

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    // explict关键字，作用：禁止编译器进行隐式转换
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 槽函数：接收线程传回的 QImage 并更新 UI
    void updateOriLabel(QImage img);
    void updateProLabel(QImage img);

private:
    Ui::MainWindow *ui;
    ThreadPool* m_pool;
    VideoWorker *worker;
    std::future<void> workerFuture;
};
#endif


// #ifndef MAINWINDOW_H
// #define MAINWINDOW_H

// #include <QMainWindow>
// #include <QImage>
// #include <atomic>
// #include <opencv2/opencv.hpp>
// #include "ultralytics_yolo11.hpp" // yolo目标检测
// #include "ThreadPool.h" // 线程池
// #include <ThreadSafeQueue.h> // 线程安全队列

// class VideoWorker : public QObject {
//     Q_OBJECT
// public:
//     VideoWorker(const std::string& url, const std::string& model, const std::string& labels);
//     ~VideoWorker() = default;

//     void process();
//     void stop();

// signals:
//     // frameReadyLeft: 发送原始画面
//     // frameReadyRight: 发送处理后(带框)的画面
//     void frameReadyLeft(QImage img);
//     void frameReadyRight(QImage img);

// private:
//     // 生产者逻辑：拉流与解码
//     void captureLoop();
//     // 消费者逻辑：退了和绘图
//     void inferenceLoop();

//     std::string rtsp_url;
//     std::string model_path;
//     std::string label_file;
//     std::atomic<bool> is_running{false};

//     // 核心缓冲区：限制大小为1或2，确保实时性
//     ThreadSafeQueue<cv::Mat> frame_queue{1};

// };

// QT_BEGIN_NAMESPACE
// namespace Ui { class MainWindow; }
// QT_END_NAMESPACE

// class MainWindow : public QMainWindow {
//     Q_OBJECT
// public:
//     // explict关键字，作用：禁止编译器进行隐式转换
//     explicit MainWindow(QWidget *parent = nullptr);
//     ~MainWindow();

// // private slots:
//     // 槽函数：接收线程传回的 QImage 并更新 UI
//     // void updateOriLabel(QImage img);
//     // void updateProLabel(QImage img);

// private:
//     Ui::MainWindow *ui;
//     ThreadPool* m_pool;
//     VideoWorker *worker;
//     std::future<void> workerFuture;
// };
// #endif