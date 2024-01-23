#ifndef CONFIGURE_HPP
#define CONFIGURE_HPP

#include <QString>

class Configure {
 public:
  Configure();
  ~Configure();
  float point_distance = 5;
  float V_rate = 4;
  QString last_open = ".";
  int last_pic = 1;
  int last_mode = 0;
  short auto_enhance_V = 1;
};

#endif CONFIGURE_HPP