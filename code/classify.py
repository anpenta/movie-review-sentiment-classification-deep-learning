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

# Classify Module
# Module to classify the sentiment of movie reviews using a FastText model.

import nltk
import torch
import torchtext

import utility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_arguments = utility.parse_input_arguments(module="classify")

# Initialize the field and dataset, build the vocabulary, and initialize the iterator.
nltk.download("punkt")
text_field = torchtext.data.Field(tokenize=nltk.word_tokenize, preprocessing=lambda x: list(nltk.bigrams(x)),
                                  fix_length=500)
dataset = torchtext.data.TabularDataset(input_arguments.movie_review_path, format="csv", skip_header=True,
                                        fields=[("review", text_field)])
text_field.build_vocab(dataset, max_size=25000)
iterator = torchtext.data.Iterator(dataset, batch_size=1000, shuffle=False, sort_key=lambda x: len(x.review),
                                   device=device)

# Load the model and make the predictions.
fasttext = torch.load(input_arguments.model_path).to(device)
predictions = utility.predict_sentiments(fasttext, iterator)

analyzed_predictions = utility.analyze_predictions(predictions)
utility.save_dataframe(analyzed_predictions, input_arguments.output_path, "analyzed_predictions")
