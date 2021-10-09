#pragma once
#include "model.h"
class ModelLoader {
 public:
  /**
   * Default constructor for ModelLoader objects.
   */
  ModelLoader();

  /**
   * Constructor for ModelLoader objects that takes image size as parameter.
   */
  ModelLoader(Model model, size_t image_size);

  /**
   * Model instance to add data to.
   */
  Model model;

  /**
   * Function that gets the model instance being filled with data.
   */
  Model &GetModel();

  /**
   * Function that gets the label number.
   */
  size_t GetLabelNumber();

  /**
   * Function that set the label number.
   */
  void SetLabelNumber(size_t new_label_number);

 private:
  /**
   * Overloaded insertion function for reading in a saved model.
   */
  friend std::ifstream& operator>>(std::ifstream& is, ModelLoader &model_loader);


  /**
   * Function that keeps track of row/col for each image in model.
   */
  void IncrementRowCol(Model &model, int &row, int &col,
                       bool &assign);


  size_t image_size;

  size_t label_number = 10;

};
