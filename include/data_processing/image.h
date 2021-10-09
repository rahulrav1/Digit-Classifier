#pragma once

#include <string>
#include <iostream>
#include <vector>

class Image {
 public:
  /**
   * Default constructor for Image object.
   */
  Image();

  /**
   * Constructor that takes vector of strings
   * representing image in training data.
   */
  Image(const std::vector<std::string>& image_);

  /**
   * Overloaded insertion operator in order to
   * read in training image data for model.
  */
  friend std::ifstream& operator>>(std::ifstream& is, Image& image);

  /**
   * Getter that returns vector of strings
   * corresponding to image in training data.
  */
  const std::vector<std::string> &GetImageString();

  const size_t GetImageSize();

  void SetImageString(std::vector<std::string> new_vector_image);

  void SetImageSize(size_t new_size);



 private:
  /**
   * Vector that acts as member variable for
   * all Image objects. Represents image from
   * training data as strings.
  */
  std::vector<std::string> vector_image_;

  /**
   * size_t representing the size of the image
   */
  size_t image_size = 28;
};
