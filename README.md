# Movie Review Sentiment Classification Deep Learning

This repository contains a deep learning system that performs sentiment classification on movie reviews using a FastText neural network. The dataset url is provided below.

Dataset url: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

## Installation

It is recommended to install conda and then create an environment for the system using the ```environment.yaml``` file. A suggestion on how to install the system and activate the environment is provided below.

```bash
git clone https://github.com/anpenta/movie-review-sentiment-classification-deep-learning.git
cd movie-review-sentiment-classification-deep-learning
conda env create -f environment.yaml
conda activate movie-review-sentiment-classification-deep-learning
```

## Running the system

To run the system for training you can provide commands through the terminal using the ```train``` module. An example is given below.

```bash
python3 train.py ./movie-review-data.csv 30 100 ./output
```
This will train a model using data from the ```./movie-review-data.csv``` file for 30 epochs with a batch size of 100 and save the training plots and the trained model in the ```./output``` directory. An example of how to see the parameters for training is provided below.

```bash
python3 train.py --help
```

To run the system for classification you can provide commands through the terminal using the ```classify``` module. An example is given below.


```bash
python3 classify.py ./movie-review-data.csv ./fasttext.pt ./output
```
This will classify the sentiment of the movie reviews from the ```./movie-review-data.csv``` file using the FastText neural network from the ```./fasttext.pt``` file and save the predictions in the ```./output``` directory. An example of how to see the parameters for classification is provided below.

```bash
python3 classify.py --help
```

## Results

As an example, below are the training results we get after training a model for 60 epochs with a batch size of 100. The model starts overfitting after about 30 training epochs.

<p float="left">
<img src=./training-results/fasttext-training-accuracy.png height="320" width="420">
<img src=./training-results/fasttext-training-loss.png height="320" width="420">
</p>

## Sources
* Joulin, Armand, et al. "Bag of tricks for efficient text classification." arXiv preprint arXiv:1607.01759 (2016).
* Maas, Andrew L., et al. "Learning word vectors for sentiment analysis." Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1. Association for Computational Linguistics, 2011.
