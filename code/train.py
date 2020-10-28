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

# Train Module
# Module to train a FastText model on movie review data.

import nltk
import torch
import torch.nn as nn
import torchtext

import fasttext
import utility

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_arguments = utility.parse_input_arguments(module="train")

# Initialize the fields and dataset, and build the vocabularies.
nltk.download("punkt")
text_field = torchtext.data.Field(tokenize=nltk.word_tokenize, preprocessing=lambda x: list(nltk.bigrams(x)),
                                  fix_length=500)
label_field = torchtext.data.LabelField(dtype=torch.float)
dataset = torchtext.data.TabularDataset(input_arguments.movie_review_path, format="csv", skip_header=True,
                                        fields=[("review", text_field), ("sentiment", label_field)])
text_field.build_vocab(dataset, max_size=25000)
label_field.build_vocab(dataset)

# Split the dataset to training set, validation set, and test set, and initialize the iterators.
training_set, test_set, validation_set = dataset.split(split_ratio=[0.5, 0.1, 0.4])
training_iterator, validation_iterator, test_iterator = torchtext.data.Iterator.splits(
  (training_set, validation_set, test_set), batch_size=input_arguments.batch_size, shuffle=True,
  sort_key=lambda x: len(x.review), device=device)

# Initialize the model, loss function, and optimizer.
fasttext = fasttext.FastText(len(text_field.vocab), 30, 1).to(device)
loss_function = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(fasttext.parameters())

# Train the model on the training data and evaluate it on the validation data. Save the model and the
# resulting plots in the output path.
training_losses, training_accuracies, validation_losses, validation_accuracies = [], [], [], []
for epoch in range(1, input_arguments.epochs + 1):
  training_results = utility.run_epoch(fasttext, training_iterator, loss_function, optimizer)
  training_losses.append(training_results[0])
  training_accuracies.append(training_results[1])

  validation_results = utility.run_epoch(fasttext, validation_iterator, loss_function)
  validation_losses.append(validation_results[0])
  validation_accuracies.append(validation_results[1])

  print("Epoch: {} | Training Loss: {:.4f} | Training Accuracy: {:.2f} | Validation Loss: {:.4f} |"
        " Validation Accuracy: {:.2f}".format(epoch, training_losses[-1], training_accuracies[-1],
                                              validation_losses[-1], validation_accuracies[-1]))
utility.save_training_plots(input_arguments.output_path, training_losses, training_accuracies, validation_losses,
                            validation_accuracies)
utility.save_model(fasttext, input_arguments.output_path)

# Evaluate the model on the test data and save the results.
test_loss, test_accuracy = utility.run_epoch(fasttext, test_iterator, loss_function)
utility.save_test_results(input_arguments.output_path, test_loss, test_accuracy)
