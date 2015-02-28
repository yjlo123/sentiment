import csv
import sys
import params
import collections
from textblob import TextBlob
from content_normalizer import ContentNormalizer

# Define the Feature tuple
Feature = collections.namedtuple ("Feature", "feature sentiment")

class FeatureExtractor:
	def __init__ (self):
		self.__feature_list = []
		self.__negative_list = []
		self.__positive_list = []

	def load_feature_list (self, filename):
		self.__feature_list = []
		self.__negative_list = []
		self.__positive_list = []

		# Load the train database into memory
		f_input = open (filename, 'rU')
		reader = csv.reader (f_input)
		next (reader)

		for row in reader:
			# Extract the feature and setiment
			feature = ContentNormalizer.normalize_content (TextBlob (row[0].strip ()))
			sentiment = ""
			if (TextBlob (row[2].strip ()).lower () == "positiv"):
				sentiment = "positive"
			elif (TextBlob (row[3].strip ()).lower () == "negativ"):
				sentiment = "negative"

			if (feature != "" and sentiment != ""):
				# Append to the feature list
				self.__feature_list.append (Feature (feature, sentiment))
				if (sentiment == "positive"):
					self.__positive_list.append (feature)
				else:
					self.__negative_list.append (feature)

	def is_feature_loaded (self):
		return len (self.__feature_list) > 0

	def extract_feature (self, document):
		if self.is_feature_loaded():
			extracted = {}

			processed = ContentNormalizer.normalize_content (document)
			tokens = processed.tokens

			for row in self.__feature_list:
				if row.feature in tokens:
					extracted[row.feature] = True
				else:
					extracted[row.feature] = False

			return extracted
		else:
			print "No feature was loaded"
			return {}