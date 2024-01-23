#include "labeldialog.h"
#include <QDebug>
#include <QObject>
#include <iostream>
#include "drawonpic.h"
#include "ui_labeldialog.h"

int last_tag_id = 0, last_color_id = 0;
static const QString Armor_names[] = {"Sentry",     "Hero",       "Engineer",
                                      "3-Infantry", "4-Infantry", "5-Infantry",
                                      "Outpost",    "Base-small", "Base-big",
                                      "3-Balance",  "4-Balance",  "5-Balance"};
static const QString Engineer_names[] = {"R-tag", "Top", "Bottom", "Entrance",
                                         "Arrow"};
static const QString Wind_names[] = {"Dark", "To-hit", "Hit"};
static const QString Color_names[] = {"Blue", "Red", "Extinguish", "Purple"};
static const QString Radar_cars_names[] = {"BlueCar",       "BlueSentry",
                                           "RedCar",        "RedSentry",
                                           "ExtinguishCar", "ExtinguishSentry"};

LabelDialog::LabelDialog(QVector<box_t>::iterator box_iter,
                         const LabelMode mode,
                         QWidget* parent)
    : QDialog(parent), ui(new Ui::LabelDialog), current_box(box_iter) {
  ui->setupUi(this);
  setup_boxes(mode);
  ui->class_box->setCurrentIndex(current_box->tag_id);
  ui->color_box->setCurrentIndex(current_box->color_id);
  // ui->color_box->setCurrentText(current_box->getName());
  QWidget::connect(ui->cancelButton, &QPushButton::pressed, this,
                   &QDialog::hide);
}

LabelDialog::~LabelDialog() {
  delete ui;
}

void LabelDialog::on_saveButton_pressed() {
  current_box->set_class(ui->color_box->currentIndex(),
                         ui->class_box->currentIndex());
  last_tag_id = current_box->tag_id;
  last_color_id = current_box->color_id;
  emit changeBoxEvent();
  this->hide();
}

void LabelDialog::on_deleteButton_pressed() {
  emit removeBoxEvent(current_box);
  this->hide();
}

void update_list_name(const LabelMode mode) {
  switch (mode) {
    case Armor:
      for (short i = 0; i < 12; ++i)
        tag_name[i] = Armor_names[i];
      break;
    case Wind:
      for (short i = 0; i < 3; ++i)
        tag_name[i] = Wind_names[i];
      break;
    case Wind_Armor:
      tag_name[0] = "Wind";
      break;
    case Engineer:
      for (short i = 0; i < 5; ++i)
        tag_name[i] = Engineer_names[i];
      break;
    case Radar_cars:
      for (short i = 0; i < 6; ++i)
        tag_name[i] = Radar_cars_names[i];
      break;
  }
}

void LabelDialog::setup_boxes(const LabelMode mode) {
  ui->color_box->clear();
  ui->class_box->clear();
  switch (mode) {
    case Armor:
      for (auto& name : Armor_names)
        ui->class_box->addItem(name);
      for (auto& name : Color_names)
        ui->color_box->addItem(name);
      break;
    case Wind:
      for (auto& name : Wind_names)
        ui->class_box->addItem(name);
      ui->color_box->addItem(Color_names[0]);
      ui->color_box->addItem(Color_names[1]);
      break;
    case Wind_Armor:
      ui->class_box->addItem("Wind");
      ui->color_box->addItem(Color_names[0]);
      ui->color_box->addItem(Color_names[1]);
      break;
    case Engineer:
      for (auto& name : Engineer_names)
        ui->class_box->addItem(name);
      ui->color_box->addItem("Gold");
      break;
    case Radar_cars:
      for (auto& name : Radar_cars_names)
        ui->class_box->addItem(name);
      break;
  }
}