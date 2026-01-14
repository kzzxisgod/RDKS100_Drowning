#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "gflags/gflags.h" // 解析命令行库
#include "sp_codec.h" // 地平线硬件解码库
#include "common_utils.hpp"

// 定义命令行参数
// 海康相机的rtsp流，将网段从137修改到127是为了解决和电脑网段冲突
DEFINE_string(rtsp_url, "rtsp://admin:waterline123456@192.168.127.15", "RTSP URL");
DEFINE_string(model_path, "/home/sunrise/Desktop/test_rtsp/ultralytics_YOLO.hbm", "YOLO11 model path"); 
DEFINE_string(label_file, "/home/sunrise/Desktop/qt_rtsp/classes.names", "Label file path");
DEFINE_double(score_thres, 0.25, "Confidence threshold"); // 置信度阈值，略去低于这个阈值的检测框
DEFINE_double(nms_thres, 0.45, "NMS threshold"); // 交并比IoU阈值
// 非极大值抑制，清除冗余检测框的算法，在运行目标检测模型的时候，会在同一物体周围生成多个重叠的候选框，这个阈值的作用就是从这些重叠框中选出最准确的一个，并删除其他的
// 工作逻辑：计算两个候选框的重叠面积，如果重叠比例超过了上面设定的阈值0.45，就会认为这两个框实在检测同一个物体，从而删除得分较低的那个框

// 构造函数
VideoWorker::VideoWorker(const std::string& url, const std::string& model, const std::string& labels)
    : rtsp_url(url)
    , model_path(model)
    , label_file(labels)
    , is_running(true) 
{}

/**
 * @brief 在 Mat 图像上绘制检测框
 */
void draw_detections(cv::Mat &img, const std::vector<Detection> &res, const std::vector<std::string> &labels) {
    for (const auto &det : res) {
        // 获取矩形坐标
        cv::Rect box(static_cast<int>(det.bbox[0]), 
                     static_cast<int>(det.bbox[1]), 
                     static_cast<int>(det.bbox[2] - det.bbox[0]), 
                     static_cast<int>(det.bbox[3] - det.bbox[1]));

        // 绘制矩形框
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        
        // 绘制标签
        if (det.class_id < (int)labels.size()) {
            std::string label_text = labels[det.class_id] + " " + std::to_string(det.score).substr(0, 4);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            // 绘制标签背景板
            cv::rectangle(img, cv::Point(box.x, box.y - labelSize.height),
                          cv::Point(box.x + labelSize.width, box.y + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
            // 绘制文字
            cv::putText(img, label_text, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
}

void VideoWorker::process() {
    is_running = true;

    // 开启推理消费线程
    std::thread consumer_thread(&VideoWorker::inferenceLoop, this);

    // 当前线程执行生产逻辑（拉流解码）
    captureLoop();

    if (consumer_thread.joinable()) {
        consumer_thread.join();
    }
}

// 生产者：负责地平线硬件解码
void VideoWorker::captureLoop() {
    void* decoder = sp_init_decoder_module();
    const int W = 1920, H = 1080;
    sp_start_decode(decoder, const_cast<char*>(rtsp_url.c_str()), 0, SP_ENCODER_H264, W, H);

    cv::Mat yuv(H * 3 / 2, W, CV_8UC1);
    while (is_running) {
        if (sp_decoder_get_image(decoder, reinterpret_cast<char*>(yuv.data)) != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

        // 入队：如果队列满，ThreadSafeQueue 会自动丢弃最旧帧
        frame_queue.enqueue(bgr.clone()); 
    }

    sp_stop_decode(decoder);
    sp_release_decoder_module(decoder);
}

// 消费者：负责 BPU 推理与 UI 更新
void VideoWorker::inferenceLoop() {
    YOLO11 yolo11(model_path);
    auto class_names = load_linewise_labels(label_file);

    while (is_running) {
        // 阻塞等待新帧
        cv::Mat frame = frame_queue.dequeue(); 
        
        // 1. 发送原始画面
        emit frameReadyLeft(QImage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888).copy());

        // 2. BPU 推理：此时解码线程已在并行准备下一帧
        yolo11.pre_process(frame);
        yolo11.infer();
        auto results = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, 1920, 1080);

        // 3. 绘制并发送
        cv::Mat resMat = frame.clone();
        draw_detections(resMat, results, class_names);
        emit frameReadyRight(QImage(resMat.data, resMat.cols, resMat.rows, resMat.step, QImage::Format_BGR888).copy());
    }
}

void VideoWorker::stop() {
    is_running = false;
}

MainWindow::MainWindow(QWidget *parent) 
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 1. 初始化线程池
    m_pool = new ThreadPool(4); // 创建一个包含4个线程的线程池
    worker = new VideoWorker(FLAGS_rtsp_url, FLAGS_model_path, FLAGS_label_file);

    // 设置 Label 自动缩放以适应窗口
    // ui->Ori_Label->setScaledContents(true);
    // ui->Pro_Label->setScaledContents(true);

    // 连接信号槽
    connect(worker, &VideoWorker::frameReadyLeft, this, &MainWindow::updateOriLabel);
    connect(worker, &VideoWorker::frameReadyRight, this, &MainWindow::updateProLabel);
    
    // 提交任务到线程池执行 process 函数
    // std::bind和lambda表达式
    // 在c++11之后，lambda表达式在大多数情况下已经取代了std::bind
    // workerFuture = pool.submitTask(std::bind(&VideoWorker::process, worker));
    m_pool->enqueue([this]() { worker->process(); });
}

void MainWindow::updateOriLabel(QImage img) {
    ui->Ori_Label->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::updateProLabel(QImage img) {
    ui->Pro_Label->setPixmap(QPixmap::fromImage(img));
}

MainWindow::~MainWindow() {
    if (worker) {
        worker->stop();
        if (workerFuture.valid()){
            workerFuture.get();
        }
        delete worker;
    }
    delete ui;
}