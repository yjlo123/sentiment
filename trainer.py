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

# Define the record tuple
Record = collections.namedtuple ("Record", "content sentiment")

class Trainer:
	def __init__ (self):
		self.__train_data = []
		self.__feature_extractor = FeatureExtractor ()

	# Read the database file into the train_data field as a list of Record tuple
	def __load_database (self, filename, train_size):
		# Load the train database into memory
		self.__train_data = []
		f_input = open (filename, 'rt')
		reader = csv.reader (f_input)
		next (reader)

		record_read = 0
		for row in reader:
			# Extract the content and setiment
			content = TextBlob (row[3].strip ())
			sentiment = TextBlob (row[4].strip ())

			# Append to the train_data
			self.__train_data.append (Record (content, sentiment))

			# Count the record read, stop if the desired has been reached
			record_read = record_read + 1
			if train_size >= 0 and record_read >= train_size:
				break

		# Process the data
		self.__process_data ()

	# Process the raw data from user
	def __process_data (self):
		# Remove all irrelevant record
		self.__train_data = [row for row in self.__train_data if row.sentiment != "irrelevant"]

		temp = []
		# Tokenize, stem, and remove all unnecessary information 
		for row in self.__train_data:
			content = ContentNormalizer.normalize_content (row.content)
			sentiment = row.sentiment

			# Skip the row if the content is shorter than 3 characters
			if (len (content) > 3):
				temp.append (Record (content, sentiment))

		self.__train_data = temp

	# Train the classifier using the database provided
	def train_classifier (self, filename, train_size):
		#print "Training classifier..."
		self.__load_database (filename, train_size)
		self.__classifier = NaiveBayesClassifier (self.__train_data)

	def train_analyzer (self, filename):
		#print "Loading feature list..."
		self.__feature_extractor.load_feature_list (filename)

	def classify (self, content):
		processed = ContentNormalizer.normalize_content (content)

		classification = self.__classifier.prob_classify (processed)
		
		result = classification.max ()
		max_prob = classification.prob (result)

		if (result != "neutral"):
			for strength_lvl in range (0, len (params.STRENGTH_THRESHOLD)):
				if max_prob < params.STRENGTH_THRESHOLD[strength_lvl]:
					if strength_lvl == 0:
						result = "irrelevant"
					else:
						result = (result, strength_lvl)
					break
		else:
			result = (result, 0)
		return (result, max_prob)


	# Return the classifier that is trained by the data provided
	def get_trained_classifier (self):
		return self.__classifier

	# Return the train data
	def get_trained_data (self):
		return self.__train_data




'''
#### Main Program

# Get train data and train the classifier
trainer = Trainer ()
trainer.train_classifier (params.TRAINER_PARAM_INPUT_FILE_NAME, params.TRAINER_PARAM_TRAIN_SIZE)
trainer.train_analyzer (params.FEATURE_FILE_NAME)

print (trainer.classify ("Rep. Weiner exposes yet another republican lie, while Virginia Foxx squeaks"))
print (trainer.classify ("all LIES from the republican"))
print (trainer.classify ("sky is grey, i just want to sleep"))
print (trainer.classify ("how are you today"))
'''
