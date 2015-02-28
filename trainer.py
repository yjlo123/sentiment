from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import PunktWordTokenizer
from nltk.stem import PorterStemmer
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

	# Remove all URL in the content string
	@staticmethod
	def __remove_url (content):
		return TextBlob (re.sub(r'^https|http?:\/\/.*[\r\n]*', '', "{0}".format (content), flags=re.MULTILINE))

	# Tokenize the content string into a list of tokens
	@staticmethod
	def __tokenize (content):
		tokenizer = PunktWordTokenizer ()
		return content.tokenize (tokenizer)

	# Remove all @ tag in the token list
	@staticmethod
	def __remove_ad_tag (tokens):
		temp = []
		for i in range (0, len (tokens)):
			if tokens[i] == "@": continue				# Do not copy the @
			if i > 0 and tokens[i-1] == "@": continue	# Do not copy anything right after @
			temp.append (tokens[i])
		return temp

	# Remove all # tag in the token list
	@staticmethod
	def __remove_hash_tag (tokens):
		temp = []
		for i in range (0, len (tokens)):
			if tokens[i] == "#": continue				# Do not copy the #
			if i > 0 and tokens[i-1] == "#": continue	# Do not copy anything right after #
			temp.append (tokens[i])
		return temp

	# Remove all stopwords in the token list
	@staticmethod
	def __remove_stopwords (tokens, stopword_list):
		return [token for token in tokens if token not in stopword_list]

	# Get rid of all non-alphabet tokens
	@staticmethod
	def __remove_nonalphabet (tokens):
		return [token for token in tokens if token.isalpha ()]

	# Stem the token into the normalized from
	@staticmethod
	def __stem_tokens (tokens):
		stemmer = PorterStemmer ()
		return [stemmer.stem_word (token) for token in tokens]

	# Concatinate all tokens into a single string
	@staticmethod
	def __join_tokens (tokens):
		content = TextBlob ("")
		for token in tokens:
			content = content + " " + token
		return content

	# Normalize a single content from user
	@staticmethod
	def normalize_content (content):
		content = content.lower ()
		content = Trainer.__remove_url (content)
		tokens = Trainer.__tokenize (content)
		tokens = Trainer.__remove_ad_tag (tokens)
		tokens = Trainer.__remove_hash_tag (tokens)
		tokens = Trainer.__remove_nonalphabet (tokens)
		tokens = Trainer.__stem_tokens (tokens)
		tokens = Trainer.__remove_stopwords (tokens, stopword_list.get_stopwords())
		content = Trainer.__join_tokens (tokens)

		return content

	# Process the raw data from user
	def __process_data (self):
		# Remove all irrelevant record
		self.__train_data = [row for row in self.__train_data if row.sentiment != "irrelevant"]

		temp = []
		# Tokenize, stem, and remove all unnecessary information 
		for row in self.__train_data:
			content = Trainer.normalize_content (row.content)
			sentiment = row.sentiment

			# Skip the row if the content is shorter than 3 characters
			if (len (content) > 3):
				temp.append (Record (content, sentiment))

		self.__train_data = temp

	# Train the classifier using the database provided
	def train_classifier (self, filename, train_size):
		self.__load_database (filename, train_size)
		self.__classifier = NaiveBayesClassifier (self.__train_data)

	def classify (self, content):
		processed = Trainer.normalize_content (content)

		classification = self.__classifier.prob_classify (processed)
		confident = classification.prob (classification.max ())

		return confident

	# Return the classifier that is trained by the data provided
	def get_trained_classifier (self):
		return self.__classifier

	# Return the train data
	def get_trained_data (self):
		return self.__train_data




#### Main Program

# Get train data and train the classifier
trainer = Trainer ()
trainer.train_classifier (params.TRAINER_PARAM_INPUT_FILE_NAME, params.TRAINER_PARAM_TRAIN_SIZE)

print (trainer.classify ("Rep. Weiner exposes yet another republican lie, while Virginia Foxx squeaks"))
print (trainer.classify ("all LIES from the republican"))
