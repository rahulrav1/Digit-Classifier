#pragma once
#include "image.h"
#include <vector>
#include <utility>
#include <fstream>
#include <unordered_map>
class Model {
 public:
  /**
  * Default constructor for Model objects.
  */
  Model();

  /**
  * Constructor that takes path for data labels.
  */
  Model(const std::string labels_path);

  /**
  * Constructor that takes path for data labels.
  */
  Model(const Model &new_model);

  /**
   * Function add_image that adds an Image object
   * to a vector of images.
  */
  void add_image(Image img);

  /**
  * Function that computes the number
  * of colored squares for all i, j
  * for all classes of images.
  */

  void calculate_occurrences();

  /**
  * Function that preprocesses all labels in training data.
  */
  void ReadLabels(const std::string &file_name_);

  /**
  * Function that computes probabilities
  * for each cell i, j in the training dataset.
  */
  void compute_probabilities();

  /**
  * Function that returns the vector of training images.
  */
  const std::vector<Image> &GetTrainingImages() const;

  /**
  * Function that returns the vector of training labels.
  */
  const std::vector<int> &GetTrainingLabels() const;

  /**
  * Function that returns the occurrences for all classes.
  */
  const std::unordered_map<int, int> &GetClassOccurrences() const;

  /**
  * Function that returns the shaded occurrences for all images.
  */
  const std::unordered_map<int, std::vector<std::vector<int>>> &GetShadedOccurrences() const;

  /**
  * Function that returns the probabilities for each class.
  */
  const std::unordered_map<int, double> &GetClassProbabilities() const;

  /**
  * Function that returns the shaded probabilites for each class.
  */
  const std::unordered_map<int, std::vector<std::vector<double>>> &GetShadedProbabilities() const;

  /**
  * Function that returns the number of labels in labels dataset.
  */
  int GetLabelsNumber();

  /**
  * Function that returns the size of images in training data.
  */
  size_t GetImageSize();

  /**
  * Function that sets the path for training data for the model.
  */
  void SetFilePath(std::string new_path);

  /**
  * Function that sets the image size for model.
  */
  void SetImageSize(size_t new_image_size);


  /**
  * Function that sets the label number for model.
  */
  void SetLabelNumber(size_t new_labels_number);


  /**
  * Function that sets class probabilities.
  */
  void SetClassProbabilities(std::unordered_map<int, double> new_probabilities);

  /**
  * Function that sets shaded probabilities.
  */
  void SetShadedProbabilities(std::unordered_map<int, std::vector<std::vector<double>>> new_probabilities);

 private:
  /**
  * Data member representing the images in training data.
  */
   std::vector<Image> training_data_;

   /**
   * Data member representing the labels in training data.
   */
   std::vector<int> labels_;

   /**
   * Data member representing the number of occurrences of a certain class
   * in the training data..
   */
   std::unordered_map<int, int> class_occurrences_;

   /**
   * Data member representing the number of shaded occurrences for each
   * cell i,j for each class.
   */
   std::unordered_map<int, std::vector<std::vector<int>>> shaded_occurrences_;

   /**
   * Data member representing the class probabilities for each class.
   */
   std::unordered_map<int, double> class_probabilities_;

   /**
   * Member variable representing the shaded probabilities in training data.
   */
   std::unordered_map<int, std::vector<std::vector<double>>> shaded_probabilities_;

   /**
   * Overloaded insertion function for reading model.
   */
   friend std::ifstream& operator>>(std::ifstream& is, Model &model);

   /**
   * Overloaded extraction function for saving model.
   */
   friend std::ofstream & operator << (std::ofstream &out, Model &model);

   /**
   * Function that computes class probabilities.
   */
   void ComputeClassProbabilities();

   /**
   * Function that computes pixel shaded/unshaded probabilities.
   */
   void ComputePixelProbabilities();

   /**
   * Member variable string representing path for training labels.
   */
   std::string file_path_ = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/traininglabels";

   /**
   * Member variable size_t representing the dimensions of an image.
   */
   size_t image_size_;

   /**
   * Member variable int representing the number of classes for an image.
   */
   int number_labels_ = 10;

   /**
   * Member variable to get rid of magic number.
   */
   const int kLaplaceMultiplier = 2;

   /**
   * Member variable representing the laplace value k used when computing probabilities.
   */
   const int kLaplace = 1;
};