#include "data_processing/model_loader.h"
#include <string>
ModelLoader::ModelLoader() {

}
ModelLoader::ModelLoader(Model new_model, size_t model_image_size) {
  image_size = model_image_size;
  new_model.SetImageSize(image_size);
}
std::ifstream& operator>>(std::ifstream& is, ModelLoader &model_loader) {
  // Initializing variables

  std::string expression_line;

  Model &model = model_loader.GetModel();

  std::unordered_map<int, double> class_probabilities = model.GetClassProbabilities();

  std::unordered_map<int, std::vector<std::vector<double>>> shaded_probabilities = model.GetShadedProbabilities();

  std::vector<std::vector<double>> image_vector(model_loader.image_size);

  model.SetImageSize(model_loader.image_size);

  // Initializing flags and counters

  bool loading_image = true;
  bool loading_class = false;
  bool assign = false;

  int file_line = 1;
  int image_row = 0;
  int image_col = 0;
  int current_class = 0;
  int multiplier = 2;

  int image_size = model_loader.image_size;
  size_t label_number = model_loader.GetLabelNumber();

  // Iterates through saved model file

  while(!is.eof()) {
    std::getline(is, expression_line);

    // Maps vector after storing/loading probabilities for one class
    if(assign) {
      shaded_probabilities[current_class] = image_vector;
      image_vector.clear();
      image_vector.resize(image_size);
      assign = false;
      current_class++;
    }

    // Resizes vector to initialize its capacity
    if(image_col == 0) {
      image_vector[image_row].resize(model_loader.image_size);
    }

    // Parses current pixel probability
    if(loading_image) {
      image_vector[image_row][image_col] = atof(expression_line.c_str());
      image_col++;
    }

    // Parses current class probability
    if(loading_class) {
      class_probabilities[current_class] = atof(expression_line.c_str());
      current_class++;
    }

    // Helper function to maintain counter variables
    model_loader.IncrementRowCol(model, image_row, image_col, assign);

    // Checks if we have finished loading shaded probabilities

    if(file_line == ((image_size * image_size) * label_number)) {
      loading_image = false;
    }

    // Checks if we have reached class probabilities

    if(file_line == ((image_size * image_size) * label_number) * multiplier) {
      loading_image = false;
      loading_class = true;
      current_class = 0;
    }

    file_line++;
  }

  // Reassigns probabilities for to model object
  model.SetClassProbabilities(class_probabilities);

  model.SetShadedProbabilities(shaded_probabilities);

  return is;
}
void ModelLoader::IncrementRowCol(Model &passed_model, int &row, int &col, bool &assign) {
  // Resets column to zero and increments row counters
  if(col == passed_model.GetImageSize()) {
    col = 0;
    row++;
  }
  // Resets row counter and sets assign flag to true
  if(row == passed_model.GetImageSize()) {
    row = 0;
    assign = true;
  }
}
Model &ModelLoader::GetModel() {
  return model;
}
size_t ModelLoader::GetLabelNumber() {
  return label_number;
}

void ModelLoader::SetLabelNumber(size_t new_label_number) {
  label_number = new_label_number;
}


