#pragma once
#include "model.h"
class ModelClassifier {
 public:
  /**
   * Constructor for ModelClassifier objects that takes two models.
   */
  ModelClassifier(Model training_model, Model testing_model);

  /**
   * Function to compute accuracy of the model.
   */
  double ComputeAccuracy(Model &testing_model);

  std::vector<size_t> GetClassifiedImages();
  std::vector<double> &GetLikelihoodScores();

 private:
  size_t labels_;
  double max_initializer_ = -10000000000000;

  /**
   * Vector that holds the labels associated with the ith image.
   */
  std::vector<size_t> classified_images_;

  /**
   * Vector that stores the max likelihood score for each image.
   */
  std::vector<double> likelihood_scores_;

  /**
   * Function to classify images that associates a label with each images
   */
  void ClassifyImages(Model &training_model, Model &testing_model);

  /**
   * Function that calculates the likelihood value for each image.
   */
  size_t CalculateLikelihood(Image image, std::unordered_map<int, double> class_prob,
                             std::unordered_map<int, std::vector<std::vector<double>>> shaded_prob);
};
