#include <iostream>
#include <unordered_map>
#include <istream>
#include <boost/program_options.hpp>

#include "../include/data_processing/model.h"
#include "../include/data_processing/model_loader.h"
#include "../include/data_processing/model_classifier.h"

int main(int argc, char *argv[]) {

  Model model;

  std::string train_images_path = "../data/trainingimages";

  std::string train_labels_path = "../data/traininglabels";

  std::string test_labels_path = "../data/testlabels";

  std::string test_images_path = "../data/testimages";

  std::string save_model_path = "../data/model.txt";

  std::string load_model_path = "../data/saved_model.txt";


  bool load_model = false;
  int image_size = 0;

  // Checks if the user has specificed arguments via command line
  if (argc > 1) {
    try {
      // Sets up boost and flags that can be parsed from command line
      boost::program_options::options_description desc{"Options"};
      desc.add_options()(
          "train_images",
          boost::program_options::value<std::string>()->default_value(
              train_images_path))(
          "train_labels",
          boost::program_options::value<std::string>()->default_value(
              train_labels_path))(
          "save", boost::program_options::value<std::string>()->default_value(
                      save_model_path))(
          "load", boost::program_options::value<std::string>()->default_value(
                      load_model_path))(
          "size_image", boost::program_options::value<int>()->default_value(
              image_size))(
          "test_images",
          boost::program_options::value<std::string>()->default_value(
              test_images_path))(
          "test_labels",
          boost::program_options::value<std::string>()->default_value(
              test_labels_path));

      // Parses command line arguments and stores results in variables_map
      boost::program_options::variables_map vm;
      store(parse_command_line(argc, argv, desc), vm);
      notify(vm);

      // Finds the specific arguments the user specified

      if(vm.at("train_images").as<std::string>() != train_images_path) {
        train_images_path = vm["train_images"].as<std::string>();
      }
      if(vm.at("train_labels").as<std::string>() != train_labels_path) {
        train_labels_path = vm["train_labels"].as<std::string>();
      }
      if(vm.at("save").as<std::string>() != save_model_path) {
        save_model_path = vm["save"].as<std::string>();
      }
      if(vm.at("load").as<std::string>() != load_model_path) {
        load_model_path = vm["load"].as<std::string>();
        load_model = true;
      }
      if(vm.at("size_image").as<int>() != image_size) {
        image_size = vm["size_image"].as<int>();
      }
      if(vm.at("test_images").as<std::string>() != test_images_path) {
        test_images_path = vm["test_images"].as<std::string>();
      }
      if(vm.at("test_labels").as<std::string>() != test_labels_path) {
        test_labels_path = vm["test_labels"].as<std::string>();
      }
    }
    catch(const std::exception &e) {
      std::cout << e.what();
    }
  }

  // If the user doesn't want to load a model
  // Then preprocess training data

  if(!load_model) {
    std::ifstream in(train_images_path.c_str());
    in >> model;
    model.calculate_occurrences();
    model.compute_probabilities();
    in.close();
    std::ofstream out(save_model_path.c_str());
    out << model;
    out.close();
  }

  // If the user does want to load a model
  // and if they specified an image size
  // then load the model

  else if(image_size != 0) {
    std::ifstream model_in(load_model_path.c_str());
    ModelLoader model_loader(model, image_size);
    model_in >> model_loader;
    model_in.close();
  }

  // Initialize test model path and read in test data
  Model test_model(test_labels_path.c_str());
  std::ifstream test_in(test_images_path.c_str());
  test_in >> test_model;
  test_in.close();

  // Pass training and testing models to ModelClassifier
  ModelClassifier model_classifier(model, test_model);
  test_model.SetLabelNumber(test_model.GetTrainingImages().size());

  // Compute and print the accuracy of model
  std::cout << "Accuracy: " << model_classifier.ComputeAccuracy(test_model) << std::endl;
  return 0;
}
