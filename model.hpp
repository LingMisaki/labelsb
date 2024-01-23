#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include <opencv2/opencv.hpp>
#include <QMap>
#include <QString>
#include <QPoint>
#include <QPolygon>
#include <QDebug>

extern QString tag_name[12];
extern int last_color_id, last_tag_id;

struct box_xywh {
    float x, y, w, h;

    void to_pts(QPointF *pts) {
        pts[0].rx() = x - w / 2.f;
        pts[0].ry() = y - h / 2.f;
        pts[1].rx() = x + w / 2.f;
        pts[1].ry() = y + h / 2.f;
    }
};

class box_t {
public:
    QPointF pts[5];
    int color_id = last_color_id, tag_id = last_tag_id;
    float conf = -1;

    QString getName() const {
        return tag_name[tag_id];
    }

    void set_class(const int color, const int tag) {
        color_id = color;
        tag_id = tag;
    }

    box_xywh to_xywh() {
        float x = this->pts[0].x();
        float y = this->pts[0].y();
        float w = this->pts[1].x() - this->pts[0].x();
        float h = this->pts[1].y() - this->pts[0].y();
        return {x + w / 2.f, y + h / 2.f, w, h};
    }

    QPolygonF getStandardPloygon() const {
        QPolygonF pts;
        pts.append({0., 0.});
        pts.append({0., (2 <= tag_id && tag_id <= 7) ? (725.) : (660.)});
        pts.append({(2 <= tag_id && tag_id <= 7) ? (780.) : (1180.), (2 <= tag_id && tag_id <= 7) ? (725.) : (660.)});
        pts.append({(2 <= tag_id && tag_id <= 7) ? (780.) : (1180.), 0.});
        return pts;
    }
};

struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};

class SmartModel {
public:
    explicit SmartModel();

    bool run(const QString &image_file, QVector<box_t> &boxes);
    void preprocess(const cv::Mat &img, cv::Mat &out, cv::Size size);

    QString get_mode() const { return mode; }

    PreParam param;

private:
    cv::dnn::Net net;
    QString mode;
};


#endif /* _MODEL_HPP_ */
