# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Utility Module
# Utility functions to run movie review sentiment classification.

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import torch

plt.rcParams.update({"font.size": 12})


def compute_correct_prediction_count(predictions, labels):
  correct_prediction_count = torch.round(predictions).eq(labels).sum().item()
  return correct_prediction_count


def run_epoch(model, iterator, loss_function, optimizer=None):
  # Train the model only if an optimizer was provided.
  if optimizer:
    torch.set_grad_enabled(True)
    model.train()
  else:
    torch.set_grad_enabled(False)
    model.eval()

  epoch_loss = 0
  epoch_accuracy = 0
  for batch in iterator:
    # Perform forward propagation and compute the loss.
    inputs = batch.review
    labels = batch.sentiment
    outputs = model(inputs).squeeze()
    loss = loss_function(outputs, labels)

    if optimizer:
      # Perform backward propagation and update the model's weights.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    epoch_loss += loss.item() * inputs.size(0)
    predictions = torch.sigmoid(outputs)
    epoch_accuracy += compute_correct_prediction_count(predictions, labels)

  dataset_length = len(iterator.dataset)
  epoch_loss = epoch_loss / dataset_length
  epoch_accuracy = epoch_accuracy / dataset_length
  return epoch_loss, epoch_accuracy


def save_training_plots(directory_path, training_losses, training_accuracies, validation_losses, validation_accuracies):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving training plots | Directory path: {}".format(directory_path))

  plt.plot(training_accuracies)
  plt.plot(validation_accuracies)
  plt.title("FastText accuracy")
  plt.ylabel("Accuracy")
  plt.xlabel("Epoch")
  plt.legend(["Training set", "Validation set"], loc="upper left")
  plt.savefig("{}/fasttext-training-accuracy".format(directory_path))
  plt.close()

  plt.plot(training_losses)
  plt.plot(validation_losses)
  plt.title("FastText loss")
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.legend(["Training set", "Validation set"], loc="upper left")
  plt.savefig("{}/fasttext-training-loss".format(directory_path))
  plt.close()


def save_model(model, directory_path):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving model | Directory path: {}".format(directory_path))

  torch.save(model, "{}/fasttext.pt".format(directory_path))


def save_test_results(directory_path, test_loss, test_accuracy):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving test_results | Directory path: {}".format(directory_path))

  with open("{}/fasttext-test-results.txt".format(directory_path), "w+") as output_file:
    output_file.write("Model's loss on unseen data: " + str(test_loss))
    output_file.write("\nModel's accuracy on unseen data: " + str(test_accuracy))


def predict_sentiments(model, iterator):
  print("Predicting sentiments")
  torch.set_grad_enabled(False)
  model.eval()

  predictions = torch.tensor([])
  for batch in iterator:
    inputs = batch.review
    outputs = model(inputs).squeeze()
    predictions = torch.cat((predictions, torch.sigmoid(outputs)), dim=0)

  return predictions


def analyze_predictions(predictions):
  print("Analyzing predictions")
  predictions = predictions.tolist()

  # Determine the sentiments and probabilities and store them in a dataframe. Assume that the predictions are in
  # the same order as the reviews.
  analyzed_predictions = pd.DataFrame()
  labels = ["negative", "positive"]
  probabilities = [x if x > 0.5 else 1 - x for x in predictions]
  analyzed_predictions["label"] = [labels[round(x)] for x in predictions]
  analyzed_predictions["probability"] = probabilities

  return analyzed_predictions


def save_dataframe(dataframe, directory_path, basename):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)

  filename = basename + ".csv"
  print("Saving data | Filename: {} | Directory path: {}".format(filename, directory_path))
  dataframe.to_csv("{}/{}".format(directory_path, filename), index=False)


def parse_input_arguments(module, epoch_choices=range(1, 61, 1), batch_size_choices=range(20, 101, 10)):
  parser = None
  if module == "classify":
    parser = argparse.ArgumentParser(prog="classify", usage="classifies the sentiment of the provided movie reviews"
                                                            " using the provided FastText model")
    parser.add_argument("movie_review_path", help="directory path to movie review data; data should be a csv file"
                                                  " containing a review column with the header included")
    parser.add_argument("model_path", help="directory path to FastText model")
    parser.add_argument("output_path", help="directory path to save the predictions")
  elif module == "train":
    parser = argparse.ArgumentParser(prog="train", usage="trains a FastText model on the provided movie review data")
    parser.add_argument("movie_review_path", help="directory path to training movie review data; data should be"
                                                  " a csv file containing a review column and a sentiment column"
                                                  " with the header included")
    parser.add_argument("epochs", type=int, choices=epoch_choices, help="number of training epochs")
    parser.add_argument("batch_size", type=int, choices=batch_size_choices, help="training data batch size")
    parser.add_argument("output_path", help="directory path to save the output of training")

  input_arguments = parser.parse_args()

  return input_arguments
