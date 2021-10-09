#include <catch2/catch.hpp>
#include "../include/data_processing/model.h"
#include "../include/data_processing/model_loader.h"
#include <iostream>
#include <vector>
using namespace Catch::literals;

/**
 * Tests to ensure functionality of model loader.
 * Partition tests as follows:
 * Tests for reading in model
*/

TEST_CASE("Tests for operator overloading") {
  SECTION("Tests for reading in saved model, shaded probability") {
    Model m;
    ModelLoader model_loader(m, 28);
    //model_loader.SetLabelNumber(5);
    std::string saved_model_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/saved_model.txt";
    std::ifstream input_stream(saved_model_path.c_str());
    input_stream >> model_loader;
    Model loaded_model = model_loader.GetModel();
    std::unordered_map<int, std::vector<std::vector<double>>> shaded_probability = loaded_model.GetShadedProbabilities();
    REQUIRE(shaded_probability[3].at(5).at(19) == .377778_a);
  }
  SECTION("Tests for reading in saved model, unshaded probability") {
    Model m;
    ModelLoader model_loader(m, 28);
    std::string saved_model_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/saved_model.txt";
    std::ifstream input_stream(saved_model_path.c_str());
    input_stream >> model_loader;
    Model loaded_model = model_loader.GetModel();
    std::unordered_map<int, std::vector<std::vector<double>>> shaded_probability = loaded_model.GetShadedProbabilities();
    REQUIRE(1 - shaded_probability[6].at(18).at(18) == .198807_a);
  }
  SECTION("Tests for reading in saved model, class probability") {
    Model m;
    ModelLoader model_loader(m, 28);
    std::string saved_model_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/saved_model.txt";
    std::ifstream input_stream(saved_model_path.c_str());
    input_stream >> model_loader;
    Model loaded_model = model_loader.GetModel();
    std::unordered_map<int, double> class_probabilities = loaded_model.GetClassProbabilities();
    REQUIRE(class_probabilities[4] == .106986_a);
  }
}
