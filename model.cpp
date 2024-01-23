//
// Created by xinyang on 2021/4/28.
//

#include "model.hpp"
#include <QFile>
#include <fstream>

QString tag_name[12];

template <class F, class T, class... Ts>
T reduce(F&& func, T x, Ts... xs) {
  if constexpr (sizeof...(Ts) > 0) {
    return func(x, reduce(std::forward<F>(func), xs...));
  } else {
    return x;
  }
}

template <class T, class... Ts>
T reduce_min(T x, Ts... xs) {
  return reduce([](auto a, auto b) { return std::min(a, b); }, x, xs...);
}

template <class T, class... Ts>
T reduce_max(T x, Ts... xs) {
  return reduce([](auto a, auto b) { return std::max(a, b); }, x, xs...);
}

// 判断目标外接矩形是否相交，用于nms。
// 等效于thres=0的nms。
static inline bool is_overlap(const QPointF pts1[4], const QPointF pts2[4]) {
  cv::Rect2f box1, box2;
  box1.x = reduce_min(pts1[0].x(), pts1[1].x(), pts1[2].x(), pts1[3].x());
  box1.y = reduce_min(pts1[0].y(), pts1[1].y(), pts1[2].y(), pts1[3].y());
  box1.width =
      reduce_max(pts1[0].x(), pts1[1].x(), pts1[2].x(), pts1[3].x()) - box1.x;
  box1.height =
      reduce_max(pts1[0].y(), pts1[1].y(), pts1[2].y(), pts1[3].y()) - box1.y;
  box2.x = reduce_min(pts2[0].x(), pts2[1].x(), pts2[2].x(), pts2[3].x());
  box2.y = reduce_min(pts2[0].y(), pts2[1].y(), pts2[2].y(), pts2[3].y());
  box2.width =
      reduce_max(pts2[0].x(), pts2[1].x(), pts2[2].x(), pts2[3].x()) - box2.x;
  box2.height =
      reduce_max(pts2[0].y(), pts2[1].y(), pts2[2].y(), pts2[3].y()) - box2.y;
  return (box1 & box2).area() > 0;
}

static inline int argmax(const float* ptr, int len) {
  int max_arg = 0;
  for (int i = 1; i < len; i++) {
    if (ptr[i] > ptr[max_arg])
      max_arg = i;
  }
  return max_arg;
}

float inv_sigmoid(float x) {
  return -std::log(1 / x - 1);
}

float sigmoid(float x) {
  return 1 / (1 + std::exp(-x));
}

SmartModel::SmartModel() {
  qDebug("initializing smart model... please wait.");
  try {
    // 首先尝试加载openvino-int8模型，并进行一次空运行。
    // 用于判断该模型在当前环境下是否可用。
    QFile xml_file(":/nn/resource/model-opt-int8.xml");
    QFile bin_file(":/nn/resource/model-opt-int8.bin");
    xml_file.open(QIODevice::ReadOnly);
    bin_file.open(QIODevice::ReadOnly);
    auto xml_bytes = xml_file.readAll();
    auto bin_bytes = bin_file.readAll();
    net = cv::dnn::readNetFromModelOptimizer(
        (uchar*)xml_bytes.data(), xml_bytes.size(), (uchar*)bin_bytes.data(),
        bin_bytes.size());
    cv::Mat input(640, 640, CV_8UC3);  // 构造输入数据
    auto x = cv::dnn::blobFromImage(input);
    net.setInput(x);
    net.forward();
    mode = "openvino-int8-cpu";  // 设置当前模型模式
    return;
  } catch (cv::Exception& e) {
    qDebug(e.what());
    // openvino int8 unavailable
  }

  // int8模型不可用，加载fp32模型
  QFile onnx_file("../resource/best-transpose.onnx");
  onnx_file.open(QIODevice::ReadOnly);
  auto onnx_bytes = onnx_file.readAll();

  net = cv::dnn::readNetFromONNX(onnx_bytes.data(), onnx_bytes.size());

  try {
    // 尝试使用openvino模式运行fp32模型
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    cv::Mat input(640, 640, CV_8UC3);
    auto x = cv::dnn::blobFromImage(input) / 255.;
    net.setInput(x);
    net.forward();
    mode = "openvino-fp32-cpu";  // 设置当前模型模式
  } catch (cv::Exception& e) {
    qDebug(e.what());
    // 无法使用openvino运行fp32模型，则使用默认的opencv-dnn模式。
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    mode = "dnn-fp32-cpu";  // 设置当前模型模式
  }
}

double get_iou(const box_t& rect1, const box_t& rect2) {
  double xx1, yy1, xx2, yy2;

  xx1 = std::max(rect1.pts[0].x(), rect2.pts[0].x());
  xx2 = std::min(rect1.pts[1].x(), rect2.pts[1].x());
  yy1 = std::max(rect1.pts[0].y(), rect2.pts[0].y());
  yy2 = std::min(rect1.pts[1].y(), rect2.pts[1].y());

  if (xx1 >= xx2 || yy1 >= yy2) {  // 如果没有重叠
    return 0;
  }

  double over_area = (xx2 - xx1) * (yy2 - yy1);  // 计算重叠面积
  double area1 = (rect1.pts[1].x() - rect1.pts[0].x()) *
                 (rect1.pts[1].y() - rect1.pts[0].y());
  double area2 = (rect2.pts[1].x() - rect2.pts[0].x()) *
                 (rect2.pts[1].y() - rect2.pts[0].y());
  double iou = over_area / (area1 + area2 - over_area);
  return iou;
}

static float clamp(float val, float min, float max) {
  return val > min ? (val < max ? val : max) : min;
}

void SmartModel::preprocess(const cv::Mat& img, cv::Mat& out, cv::Size size) {
  const float inp_h = size.height;
  const float inp_w = size.width;
  float height = img.rows;
  float width = img.cols;

  float r = std::min(inp_h / height, inp_w / width);
  int padw = std::round(width * r);
  int padh = std::round(height * r);

  cv::Mat tmp;
  if ((int)width != padw || (int)height != padh) {
    cv::resize(img, tmp, cv::Size(padw, padh));
  } else {
    tmp = img.clone();
  }
  float dw = inp_w - padw;
  float dh = inp_h - padh;

  dw /= 2.0f;
  dh /= 2.0f;
  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));

  cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                     {114, 114, 114});
  cv::dnn::blobFromImage(tmp, out, 1 / 255.f, size, cv::Scalar(), true, false,
                         CV_32F);

  this->param.ratio = 1 / r;
  this->param.dw = dw;
  this->param.dh = dh;
  this->param.height = height;
  this->param.width = width;
}

bool SmartModel::run(const QString& image_file, QVector<box_t>& boxes) {
  try {
    // 加载图片，并等比例resize为640x640。空余部分用0进行填充。
    auto img = cv::imread(image_file.toStdString());

    cv::Mat input;
    this->preprocess(img, input, {640, 640});
    //        float scale = 640.f / std::max(img.cols, img.rows);
    //        cv::resize(img, img, {(int) round(img.cols * scale), (int)
    //        round(img.rows * scale)}); cv::Mat input(640, 640, CV_8UC3, 127);
    //        img.copyTo(input({0, 0, img.cols, img.rows}));
    //        cv::imshow("result", input);
    //        cv::waitKey(0);
    //        cv::destroyWindow("result");

    // TODO: 为了兼容int8模型和fp32模型的不同输入格式而加的临时操作
    //       后续会统一两个模型的输入格式
    //        cv::Mat x;
    //        if (mode == "openvino-int8-cpu") {
    //            x = cv::dnn::blobFromImage(input);
    //        } else {
    //            cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    //            x = cv::dnn::blobFromImage(input) / 255;
    //        }
    // 模型推理
    net.setInput(input);
    auto output = net.forward();

    // 模型后处理
    QVector<box_t> before_nms;
    for (int i = 0; i < output.size[1]; i++) {
      float* result = (float*)output.data + i * output.size[2];

      box_t box;
      box.tag_id = argmax(result + 4, 2);
      box.conf = result[4 + box.tag_id];
      if (box.conf < 0.6) {
        continue;
      }

      float x = result[0] - this->param.dw;
      float y = result[1] - this->param.dh;
      float w = result[2];
      float h = result[3];

      box.pts[0].rx() =
          clamp((x - 0.5f * w) * this->param.ratio, 0.f, this->param.width);
      box.pts[0].ry() =
          clamp((y - 0.5f * h) * this->param.ratio, 0.f, this->param.height);
      box.pts[1].rx() =
          clamp((x + 0.5f * w) * this->param.ratio, 0.f, this->param.width);
      box.pts[1].ry() =
          clamp((y + 0.5f * h) * this->param.ratio, 0.f, this->param.height);

      before_nms.append(box);
    }
    std::sort(before_nms.begin(), before_nms.end(),
              [](box_t& b1, box_t& b2) { return b1.conf > b2.conf; });
    boxes.clear();
    boxes.reserve(before_nms.size());
    std::vector<bool> is_removed(before_nms.size());
    for (int i = 0; i < before_nms.size(); i++) {
      if (is_removed[i]) {
        continue;
      }
      boxes.append(before_nms[i]);
      for (int j = i + 1; j < before_nms.size(); j++) {
        if (is_removed[j]) {
          continue;
        }
        if (before_nms[j].tag_id != before_nms[i].tag_id) {
          continue;
        }
        if (get_iou(before_nms[i], before_nms[j]) > 0.6) {
          is_removed[j] = true;
        }
      }
    }

    /*
        for (auto& obj : boxes) {
          std::cout << obj.tag_id << " " << obj.conf << " " << obj.pts[0].x() <<
       " "
                    << obj.pts[0].y() << " " << obj.pts[1].x() << " "
                    << obj.pts[1].y() << std::endl;
        }

        auto res = img.clone();
        cv::Size size = {1280, 720};

        cv::resize(res, res, size);
        const std::string CLASS_NAMES[] = {"car", "sentry"};

        for (auto obj : boxes) {
          obj.pts[0].rx() *= size.width / param.width;
          obj.pts[0].ry() *= size.height / param.height;
          obj.pts[1].rx() *= size.width / param.width;
          obj.pts[1].ry() *= size.height / param.height;
          cv::rectangle(res,
                        cv::Rect2f(obj.pts[0].x(), obj.pts[0].y(),
                                   obj.pts[1].x() - obj.pts[0].x(),
                                   obj.pts[1].y() - obj.pts[0].y()),
                        {0, 255, 0}, 2);

          char text[256];
          sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.tag_id].c_str(),
                  obj.conf * 100);

          int baseLine = 0;
          cv::Size label_size =
              cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1,
       &baseLine);

          int x = (int)(obj.pts[0].x());
          int y = (int)(obj.pts[0].y()) + 1;

          cv::rectangle(
              res, cv::Rect(x, y, label_size.width, label_size.height +
       baseLine), {0, 0, 255}, -1);

          cv::putText(res, text, cv::Point(x, y + label_size.height),
                      cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
        }
        cv::imshow("result", res);
        cv::waitKey(0);
        cv::destroyWindow("result");
        */

    return true;
  } catch (std::exception& e) {
    std::ofstream ofs("warning.txt", std::ios::app);
    time_t t;
    time(&t);
    ofs << asctime(localtime(&t)) << "\t" << e.what() << std::endl;
    return false;
  }
}