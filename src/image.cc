#include "../include/data_processing/image.h"
#include <iostream>
#include <fstream>
Image::Image() {

}
Image::Image(const std::vector<std::string>& data_image) {
  vector_image_ = data_image;
}

std::ifstream& operator>>(std::ifstream& is, Image& image) {
  // Stores the current line from the file
  std::string expressionLine;

  // Vector representing one image from file
  std::vector<std::string> single_image;

  size_t row_count = 1;

  bool set_size = true;

  // Iterates to end of file
  while (!is.eof()) {

    // Gets current line of file and assigns it to expressionLine
    std::getline(is, expressionLine);

    if(set_size) {

      // Set image size to length of expression line
      image.SetImageSize(expressionLine.length());
      set_size = false;
    }
    // Checks if we have reached the end of image
    if(row_count == image.GetImageSize()) {
      single_image.push_back(expressionLine);

      image = Image(single_image);

      single_image.clear();

      row_count = 0;

      break;
    }
    else {
      // Adds line to image vector
      single_image.push_back(expressionLine);
    }
    // Increments row
    row_count++;
  }
  return is;
}
const size_t Image::GetImageSize() {
  return image_size;
}

void Image::SetImageString(std::vector<std::string> new_vector_image) {
  vector_image_ = new_vector_image;
}
void Image::SetImageSize(size_t new_size) {
  image_size = new_size;
}
const std::vector<std::string> &Image::GetImageString() {
  return vector_image_;
}
