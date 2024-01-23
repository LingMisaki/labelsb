#ifndef DRAWONPIC_H
#define DRAWONPIC_H

#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QtSvg/QSvgRenderer>
#include "configure.hpp"
#include "model.hpp"

#define NULL_IMG cv::Mat(0, 0, CV_8UC1)

enum LabelMode { Armor, Wind, Engineer, Wind_Armor, Radar_cars };

class DrawOnPic : public QLabel {
  Q_OBJECT

 public:
  explicit DrawOnPic(QWidget* parent = nullptr);

  QString model_mode() const { return model.get_mode(); }

  QString current_file;

  void reset();

  QVector<box_t>& get_current_label();

  void load_svg();

  cv::Mat modified_img = NULL_IMG, enh_img = NULL_IMG;
  bool image_equalizeHist = false;
  bool image_enhanceV = false;
  bool del_file = false;
  LabelMode label_mode = Armor;
  Configure configure;

 protected:
  void mousePressEvent(QMouseEvent* event);

  void mouseMoveEvent(QMouseEvent* event);

  void mouseReleaseEvent(QMouseEvent* event);

  void mouseDoubleClickEvent(QMouseEvent* event);

  void wheelEvent(QWheelEvent* event);

  void keyPressEvent(QKeyEvent* event);

  void paintEvent(QPaintEvent* event);

 public slots:

  void setCurrentFile(QString file);

  void loadImage();

  void saveLabel();

  void setAddingMode();

  void setNormalMode();

  void setFocusBox(int index);

  void removeBox(box_t* box);

  void smart();

  void updateBox();

  void stayPositionChanged(bool value);

  void illuminate();

  void histogram_Equalization();

  void cover_brush();

 signals:

  void labelChanged(const QVector<box_t>&);

  void delCurrentImage();

  void update_list_name_signal(const LabelMode mode);

 private:
  void loadLabel();

  void update_cover(QPointF center);

  void drawROI(QPainter& painter);

  QPointF* checkPoint();

  int label_to_size(int label, LabelMode mode);

 private:
  QSvgRenderer standard_tag_render[12];

  SmartModel model;

  QTransform norm2img;   // 归一化图像坐标到图像坐标
  QTransform img2label;  // 图像坐标到实际显示的坐标

  bool stayPosition =
      false;  //为true时加载图像时不刷新img2label（即继续显示同一局部

  // double ratio;
  // int dx, dy;
  QImage* img = nullptr;

  cv::Mat showing_modified_img;

  QPolygonF big_svg_ploygen, small_svg_ploygen, radar_car_ploygen;
  QPolygonF big_pts, small_pts, radar_pts;

  QVector<box_t> current_label;  // 归一化坐标

  QPointF* draging = nullptr;
  int cover_radius = 10;
  int focus_box_index = -1;
  int focus_point_index = -1;
  int banned_point_index = -1;
  bool F_mode = false;
  QVector<QPointF> adding;
  QPointF pos;

  QPointF right_drag_pos;
  QPointF middle_drag_pos;

  QPen pen_point_focus;
  QPen pen_point;
  QPen pen_box_focus;
  QPen pen_box;
  QPen pen_line;
  QPen pen_text;

  int latency_ms = -1;

  enum mode_t {
    NORMAL_MODE,
    ADDING_MODE,
    COVER_MODE,
  } mode = NORMAL_MODE;
};

#endif  // DRAWONPIC_H
