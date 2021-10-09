#include <catch2/catch.hpp>
#include "../include/data_processing/model.h"
#include "../include/data_processing/model_classifier.h"
#include <iostream>
#include <vector>
using namespace Catch::literals;

/**
 * Tests to ensure functionality of model classifier.
 * Tests to ensure correct likelihood score computation,
 * Tests to ensure digit is correctly classified
 * Tests to ensure accuracy is correctly computed
*/

// Tests for likelihood score computation

TEST_CASE("Tests for likelihood score computation") {
  SECTION("Test for computation on small dataset") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_test_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_test_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<double> likelihood_scores = model_classifier.GetLikelihoodScores();
    REQUIRE(likelihood_scores.at(1) == -3.061942586_a);
  }
  SECTION("Test for computation on large dataset") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/traininglabels");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/trainingimages";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/testlabels");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/testimages";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<double> likelihood_scores = model_classifier.GetLikelihoodScores();
    REQUIRE(likelihood_scores.at(4) == -98.4520818182_a);
  }
  SECTION("Test to ensure max is correct") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_test_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_test_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<double> likelihood_scores = model_classifier.GetLikelihoodScores();
    REQUIRE(*std::max_element(likelihood_scores.begin(), likelihood_scores.end()) == -2.4598825949_a);
  }
}
TEST_CASE("Tests for classifying digits") {
  SECTION("Test for classify 5") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_labels.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_images.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<size_t> classified_test_labels = model_classifier.GetClassifiedImages();
    REQUIRE(classified_test_labels[0] == 5);
  }
  SECTION("Test for classify 3") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_labels.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_images.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<size_t> classified_test_labels = model_classifier.GetClassifiedImages();
    REQUIRE(classified_test_labels[2] == 3);
  }
  SECTION("Test for classify 8") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_labels.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_images.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<size_t> classified_test_labels = model_classifier.GetClassifiedImages();
    REQUIRE(classified_test_labels[3] == 8);
  }
  SECTION("Test for classify 0") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_labels.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_images.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<size_t> classified_test_labels = model_classifier.GetClassifiedImages();
    REQUIRE(classified_test_labels[1] == 0);
  }
}
TEST_CASE("Tests for accuracy computation") {
  SECTION("Accuracy computation on small dataset") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_labels.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/training_small_images.txt";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_labels.txt");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/classifying_small_images.txt";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    std::vector<size_t> classification = model_classifier.GetClassifiedImages();
    //test_model.SetLabelNumber(5);
    double accuracy = model_classifier.ComputeAccuracy(test_model);
    REQUIRE(accuracy == 0.8);
  }
  SECTION("Accuracy computation on large dataset") {
    Model train_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/traininglabels");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/trainingimages";
    std::ifstream train_input_stream(images_path.c_str());
    train_input_stream >> train_model;
    train_input_stream.close();
    train_model.calculate_occurrences();
    train_model.compute_probabilities();
    Model test_model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/testlabels");
    images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/testimages";
    std::ifstream test_input_stream(images_path.c_str());
    test_input_stream >> test_model;
    test_input_stream.close();
    ModelClassifier model_classifier(train_model, test_model);
    //test_model.SetLabelNumber(1000);
    double accuracy = model_classifier.ComputeAccuracy(test_model);
    REQUIRE(accuracy == 0.771);
  }
}
