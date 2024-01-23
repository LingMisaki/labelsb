#include <QApplication>
#include "mainwindow.h"

int main(int argc, char* argv[]) {
  QApplication a(argc, argv);
  std::string path = "";
  int init_mode = 4;
  if (argc > 1) {
    path = argv[1];
    if (argc > 2)
      init_mode = argv[2][0] - '0';
  }
  MainWindow w(0, path, init_mode);
  w.show();

  return a.exec();
}
