#include "../include/data_processing/model_classifier.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>

ModelClassifier::ModelClassifier(Model train_model, Model test_model) {
  Model training_model_ = train_model;
  Model testing_model_ = test_model;
  // Calls ClassifyImages after object is initialized
  ClassifyImages(training_model_, testing_model_);
}

void ModelClassifier::ClassifyImages(Model &training_model, Model &testing_model) {
  // Gets required data from respective model

  std::vector<Image> testing_images = testing_model.GetTrainingImages();

  std::unordered_map<int, double> class_prob = training_model.GetClassProbabilities();

  std::unordered_map<int, std::vector<std::vector<double>>> shaded_prob = training_model.GetShadedProbabilities();

  labels_ = training_model.GetLabelsNumber();

  // Iterates through all images and calls CalculateLikelihood function

  for(Image image : testing_images) {
    size_t image_label = CalculateLikelihood(image, class_prob, shaded_prob);
    // Associates image_label with current image in vector
    classified_images_.push_back(image_label);
  }
}

size_t ModelClassifier::CalculateLikelihood(Image image, std::unordered_map<int, double> class_prob,
                                            std::unordered_map<int, std::vector<std::vector<double>>> shaded_prob) {

  std::vector<std::string> image_vector = image.GetImageString();
  size_t current_classification = 0;
  double max_likelihood = max_initializer_;

  // Iterates through all possible classes of images

  for(size_t class_index = 0; class_index < labels_; class_index++) {

    if(class_prob.find(class_index) != class_prob.end()) {

      // Initializes likelihood score to log of class probability

      double likelihood_score = log10(class_prob[class_index]);
      int row_counter = 0;

      for(std::string image_row : image_vector) {

        for(size_t index = 0; index < image_row.length(); index++) {

          if(shaded_prob[class_index].at(row_counter).at(index) != 0 ||
             (1 - shaded_prob[class_index].at(row_counter).at(index)) != 0) {

            // Checks if we are on a shaded or unshaded pixel, gets respective probability
            // and adds the log of it to likelihood score

            if(!isblank(image_row[index])) {
              likelihood_score += log10((shaded_prob[class_index]).at(row_counter).at(index));
            }

            else {
              likelihood_score += log10((1 - (shaded_prob[class_index]).at(row_counter).at(index)));
            }

          }
        }
        row_counter++;
      }

      likelihood_scores_.push_back(likelihood_score);

      // Compares likelihood score for current class to max likelihood score
      // If current score is greater we adjust max and current classification

      if(likelihood_score > max_likelihood) {
        max_likelihood = likelihood_score;
        current_classification = class_index;
      }
    }
  }
  return current_classification;
}

double ModelClassifier::ComputeAccuracy(Model &testing_model) {

  // Initializes instance variables to compute accuracy

  int correct_labels = 0;
  int total_labels = testing_model.GetTrainingImages().size();

  // Vector that stores the correct test label classifications
  std::vector<int> correct_test_labels = testing_model.GetTrainingLabels();

  for(size_t index = 0; index < classified_images_.size(); index++) {
    // Increments counter if classified label matches correct label
    if(classified_images_[index] == correct_test_labels[index]) {
      correct_labels++;
    }
  }

  // Returns percentage of correctly classified labels
  return (static_cast<double>(correct_labels)) / (static_cast<double>(total_labels));
}
std::vector<size_t> ModelClassifier::GetClassifiedImages() {
  return classified_images_;
}
std::vector<double> &ModelClassifier::GetLikelihoodScores() {
  return likelihood_scores_;
}


