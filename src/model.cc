#include "../include/data_processing/model.h"
#include <iostream>
#include <utility>
#include <vector>

Model::Model() {
  ReadLabels(file_path_);
}
Model::Model(const std::string label_path) {
  file_path_ = label_path;
  
  ReadLabels(label_path);
}
Model::Model(const Model &new_model) {
  training_data_ = new_model.GetTrainingImages();

  class_probabilities_ = new_model.GetClassProbabilities();

  shaded_probabilities_ = new_model.GetShadedProbabilities();
}

void Model::ReadLabels(const std::string &file_name) {

  // Sets up ifstream to read from specified file path
  std::ifstream in(file_name.c_str());

  std::string label_value_;

  while(std::getline(in, label_value_)) {
    labels_.push_back(std::stoi(label_value_));
  }
}
void Model::add_image(Image img) {
  // Adds an image to training data vector
  Model::training_data_.push_back(img);
}
std::ifstream& operator>>(std::ifstream& is, Model& model) {
  // Iterates through file and reads in images
  while(!is.eof()) {
    Image image;
    is >> image;
    // Adds images to training data vector
    model.add_image(image);
  }
  return is;
}
std::ofstream& operator<<(std::ofstream& os,  Model& model) {

  // Initializes all containers need to store probabilities
  std::unordered_map<int, std::vector<std::vector<double>>> stored_shaded_probabilities = model.GetShadedProbabilities();

  //std::unordered_map<int, std::vector<std::vector<double>>> stored_unshaded_probabilities = model.GetUnshadedProbabilities();
  std::unordered_map<int, double> class_probabilities = model.GetClassProbabilities();

  int total_labels = model.GetLabelsNumber();
  size_t image_dimensions = model.GetImageSize();

  // Formats output to file
  for(size_t class_index = 0; class_index < total_labels; class_index++) {

    if(stored_shaded_probabilities.find(class_index) != stored_shaded_probabilities.end()) {
      std::vector<std::vector<double>> image =
          stored_shaded_probabilities[class_index];

      for (size_t row = 0; row < image_dimensions; row++) {
        for (size_t col = 0; col < image_dimensions; col++) {

          // Loads shaded probabilities for each class
          os << image[row][col] << std::endl;
        }
      }
    }
  }


  for(size_t class_index = 0; class_index < total_labels; class_index++) {
    if(stored_shaded_probabilities.find(class_index) != stored_shaded_probabilities.end()) {
      std::vector<std::vector<double>> image =
          stored_shaded_probabilities[class_index];
      for (size_t row = 0; row < image_dimensions; row++) {
        for (size_t col = 0; col < image_dimensions; col++) {
          // Loads unshaded probabilities for each class
          os << (1 - image[row][col]) << std::endl;
        }
      }
    }
  }


  for(size_t class_index = 0; class_index < total_labels; class_index++) {
    if(class_probabilities.find(class_index) != class_probabilities.end()) {
      // Loads class probability for each class
      os << class_probabilities[class_index] << std::endl;
    }
  }

  return os;
}
void Model::calculate_occurrences() {
  int image_count = 0;
  // Finds size of image in training data (dimensions)
  image_size_ = training_data_.at(0).GetImageString().at(0).length();
  // Removes unecessary element from vector
  if(training_data_.rbegin()->GetImageString().size() == 0) {
    training_data_.pop_back();
  }

  for(Image image : training_data_) {
    int image_row_ = 0;
    int image_class_ = labels_.at(image_count);
    // Increments class occurrences for current class
    class_occurrences_[image_class_]++;
    std::vector<std::string> string_image_ = image.GetImageString();

    for(std::string pixel : string_image_) {
      for(size_t index = 0; index<pixel.length(); index++) {

        // Checks if pixel is shaded

        if(!isblank(pixel[index])) {

          std::vector<std::vector<int>> pixel_to_frequency_ = shaded_occurrences_[image_class_];

          if(pixel_to_frequency_.size() == 0) {
            pixel_to_frequency_.resize(image_size_);
          }
          if(pixel_to_frequency_[image_row_].size() == 0) {
            pixel_to_frequency_[image_row_].resize(image_size_);
          }

          // Increments current shaded pixel count
          pixel_to_frequency_[image_row_][index]++;
          // Maps updated vector
          shaded_occurrences_[image_class_] = pixel_to_frequency_;
        }
      }
      image_row_++;
    }
    image_count++;
  }
}
void Model::compute_probabilities() {
  // Computes probabilities for class and for each pixel
  ComputeClassProbabilities();
  ComputePixelProbabilities();
}
void Model::ComputeClassProbabilities() {
  // Computes probabilities for each class
  int total_class_occurrences = 0;

  for(size_t class_index = 0; class_index < number_labels_; class_index++) {
    // Finds total number of classes in data

    if(class_occurrences_.find(class_index) != class_occurrences_.end()) {
      total_class_occurrences += class_occurrences_[class_index];
    }

  }

  for(size_t class_index  = 0; class_index < number_labels_; class_index++) {

    if(class_occurrences_.find(class_index) != class_occurrences_.end()) {
      // Calculates class probability
      double class_probability = static_cast<double>(class_occurrences_[class_index] + kLaplace) /
                                 (total_class_occurrences + (number_labels_ * kLaplace));

      class_probabilities_[class_index] = class_probability;
    }

  }
}
void Model::ComputePixelProbabilities() {
  // Iterates through each class
  for(size_t class_index = 0; class_index < number_labels_; class_index++) {
    if(shaded_occurrences_.find(class_index) != shaded_occurrences_.end()) {

      std::vector<std::vector<int>> shaded_pixels = shaded_occurrences_[class_index];
      std::vector<std::vector<double>> shaded_calculations(image_size_);

      // Iterates through each vector of vectors mapped to specific class
      for(size_t row = 0; row < image_size_; row++) {
        shaded_calculations[row].resize(image_size_);
        if(shaded_pixels[row].empty()) {
          continue;
        }

        for(size_t col = 0; col < image_size_; col++) {
          int shaded_count = shaded_pixels[row][col];
          // Calculates shaded probability for pixel and class
          double shaded_probability = static_cast<double>(shaded_count + kLaplace) /
                                      ((kLaplaceMultiplier * kLaplace) + class_occurrences_[class_index]);

          // Assigns probabilities to vector of vectors
          shaded_calculations[row][col] = shaded_probability;
        }
      }
      shaded_probabilities_[class_index] = shaded_calculations;
    }
  }
}
const std::vector<Image> &Model::GetTrainingImages() const{
  return training_data_;
}
const std::vector<int> &Model::GetTrainingLabels() const {
  return labels_;
}
const std::unordered_map<int, int> &Model::GetClassOccurrences() const {
  return class_occurrences_;
}
const std::unordered_map<int, std::vector<std::vector<int>>> &Model::GetShadedOccurrences() const {
  return shaded_occurrences_;
}

const std::unordered_map<int, double> &Model::GetClassProbabilities() const {
  return class_probabilities_;
}
const std::unordered_map<int, std::vector<std::vector<double>>> &Model::GetShadedProbabilities() const {
  return shaded_probabilities_;
}
size_t Model::GetImageSize() {
  return image_size_;
}
int Model::GetLabelsNumber() {
  return number_labels_;
}

void Model::SetImageSize(size_t new_image_size) {
  image_size_ = new_image_size;
}
void Model::SetLabelNumber(size_t new_labels_number) {
  number_labels_ = new_labels_number;
}

void Model::SetClassProbabilities(std::unordered_map<int, double> new_probabilities) {
  class_probabilities_ = new_probabilities;
}
void Model::SetShadedProbabilities(std::unordered_map<int, std::vector<std::vector<double>>> new_probabilities) {
  shaded_probabilities_ = new_probabilities;
}



