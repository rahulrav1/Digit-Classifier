#include <visualizer/naive_bayes_app.h>
#include "../../include/data_processing/model_classifier.h"
#include <fstream>
namespace naivebayes {

namespace visualizer {

NaiveBayesApp::NaiveBayesApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin) {
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);

  // Preprocesses training data to classify against sketchpad drawing

  std::string images_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/trainingimages";
  std::ifstream in(images_path.c_str());
  in >> training_model;

  // Calculates occurrences and computes probabilities for training model

  training_model.calculate_occurrences();
  training_model.compute_probabilities();
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(255, 246, 148);  // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"));

  ci::gl::drawStringCentered(
      "Prediction: " + std::to_string(current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"));
}

void NaiveBayesApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_RETURN: {
      // Variables/containers used for classifying drawing
      std::vector<std::vector<bool>> pixel_states = sketchpad_.GetPixelState();
      std::string output_path = "/Users/rahulnathan/Documents/CS 126/NaiveBayes/cinder_0.9.2_mac/my-projects/naivebayes-rahulrav1/data/sketchpad_image.txt";
      std::ofstream output_stream(output_path.c_str());

      // Outputs pixel_state 2d vector to a file, this represents an image
      for (int row = 0; row < pixel_states.size(); row++) {
        for (int col = 0; col < pixel_states[row].size(); col++) {
          if (pixel_states.at(row).at(col) == 1) {
            output_stream << "#";
          } else {
            output_stream << ' ';
          }
        }
        if(row < pixel_states.size() - 1) {
          output_stream << '\n';
        }
      }

      output_stream.close();

      // Sets up streams and reads in image outputted to file in previous step
      // Computes probabilities for sketchpad image
      std::ifstream in(output_path.c_str());
      Model sketchpad_model;
      in >> sketchpad_model;
      sketchpad_model.calculate_occurrences();
      sketchpad_model.compute_probabilities();

      // Plugs training and sketchpad model into ModelClassifier instance
      ModelClassifier model_classifier(training_model, sketchpad_model);

      // Gets classified test labels and assigns predicted label to current_prediction_
      std::vector<size_t> classified_test_labels = model_classifier.GetClassifiedImages();
      current_prediction_ = classified_test_labels.at(0);

      std::cout << "Current Prediction: " << current_prediction_ << std::endl;

      break;
  }
    case ci::app::KeyEvent::KEY_DELETE:
      sketchpad_.Clear();
      break;
  }
}

}  // namespace visualizer

}  // namespace naivebayes
