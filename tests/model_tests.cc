#include <catch2/catch.hpp>
#include "../include/data_processing/model.h"
#include <iostream>
#include <vector>
using namespace Catch::literals;

/**
 * Tests to ensure functionality of model.
 * Partition tests as follows:
 * Tests for helper functions,
 * Tests for Reading/Saving Model
 * Tests for probability computations
*/

// Tests for helper functions
TEST_CASE("Tests for Helper Functions") {
  SECTION("Tests for add_image") {
    Model model;
    Image image;
    size_t old_size = model.GetTrainingImages().size();
    model.add_image(image);
    REQUIRE(old_size + 1 == model.GetTrainingImages().size());
  }
  SECTION("Tests for read_labels large dataset") {
    std::ifstream input_stream;
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/traininglabels");
    size_t actual_size = model.GetTrainingLabels().size();
    size_t expected_size = 5000;
    REQUIRE(actual_size == expected_size);
  }
}
// Tests for reading in data and saving data, includes operator overloading tests
TEST_CASE("Tests for reading/saving in training data") {
  SECTION("Tests for reading in training images") {
    Model model;
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    size_t expected_size = 5;
    REQUIRE(model.GetTrainingImages().size() == expected_size);
  }
  SECTION("Tests for read_labels small dataset") {
    std::ifstream input_stream;
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    size_t actual_size = model.GetTrainingLabels().size();
    size_t expected_size = 5;
    REQUIRE(actual_size == expected_size);
  }
  SECTION("Tests for saving model to file") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    model.compute_probabilities();
    std::string model_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_model_output.txt";
    std::ofstream output_stream(model_path.c_str());
    output_stream << model;
    std::ifstream model_reader(model_path.c_str());
    std::string first_line;
    std::getline(model_reader, first_line);
    std::string expected_line = "0.666667";
    REQUIRE(first_line.compare(expected_line) == 0);
  }
}
// Tests for probability computations on smaller images and labels datasets
TEST_CASE("Tests for probability functions/computations") {
  SECTION("Tests for calculate occurrences for class") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    std::unordered_map<int, int> class_occurrences = model.GetClassOccurrences();
    REQUIRE(class_occurrences[1] == 1);
  }
  SECTION("Tests for calculate occurrences for shaded pixel") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    std::unordered_map<int, std::vector<std::vector<int>>> shaded_occurrences = model.GetShadedOccurrences();
    REQUIRE(shaded_occurrences[2].at(0).at(0) == 1);
  }
  SECTION("Tests for calculate occurrences for unshaded pixel") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    std::unordered_map<int, int> class_occurrences = model.GetClassOccurrences();
    std::unordered_map<int, std::vector<std::vector<int>>> shaded_occurrences = model.GetShadedOccurrences();
    REQUIRE(class_occurrences[1] - shaded_occurrences[1].at(0).at(1) == 0);
  }
  SECTION("Computing class probabilities") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    model.compute_probabilities();
    std::unordered_map<int, double> class_probabilities_ = model.GetClassProbabilities();
    REQUIRE(class_probabilities_[1] == 0.1333333333_a);
  }
  SECTION("Computing shaded probabilities") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    model.compute_probabilities();
    std::unordered_map<int, std::vector<std::vector<double>>> shaded_probabilities = model.GetShadedProbabilities();
    REQUIRE(shaded_probabilities[1].at(0).at(0) == 0.6666666667_a);
  }
  SECTION("Computing unshaded probabilities") {
    Model model("/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_labels_dataset.txt");
    std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/small_images_dataset.txt";
    std::ifstream input_stream(images_path.c_str());
    input_stream >> model;
    model.calculate_occurrences();
    model.compute_probabilities();
    std::unordered_map<int, std::vector<std::vector<double>>> shaded_probabilities = model.GetShadedProbabilities();
    REQUIRE(1 - shaded_probabilities[1].at(0).at(0) == 0.3333333333_a);
  }
}









