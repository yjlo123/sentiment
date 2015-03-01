from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import PunktWordTokenizer
from nltk.stem import PorterStemmer
from content_normalizer import ContentNormalizer
from feature_extractor import FeatureExtractor
import collections
import params
import stopword_list
import sys
import csv
import re
import trainer

print "Training..."
# Get train data and train the classifier
trainer = trainer.Trainer ()
trainer.train_classifier (params.TRAINER_PARAM_INPUT_FILE_NAME, params.TRAINER_PARAM_TRAIN_SIZE)
trainer.train_analyzer (params.FEATURE_FILE_NAME)

content = raw_input(">> ")
while content != quit:
	print (trainer.classify (content))
	content = raw_input(">> ")