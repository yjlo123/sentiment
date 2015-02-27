from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import PunktWordTokenizer
from nltk.stem import PorterStemmer
import params
import stopword_list
import sys
import csv
import re

# Read the database file into an array of (content, sentiment) tuples
def load_database (filename, train_size):
	# Load the train database into memory
	train_data = []
	f_input = open (filename, 'rt')
	reader = csv.reader (f_input)
	next (reader)

	record_read = 0
	for row in reader:
		# Extract the content and setiment
		content = row[3].strip ()
		sentiment = row[4].strip ()

		# Append to the train_data
		train_data.append ((content, sentiment))

		# Count the record read, stop if the desired has been reached
		record_read = record_read + 1
		if train_size >= 0 and record_read >= train_size:
			break

	return train_data

# Remove all URL in the content
def remove_url (content):
	return re.sub(r'^https|http?:\/\/.*[\r\n]*', '', content, flags=re.MULTILINE)

# Tokenize the content
def tokenize (content):
	tokenizer = PunktWordTokenizer ()
	textblob = TextBlob (content, tokenizer = tokenizer)
	return textblob.tokens

# Remove all @ tag in the token list
def remove_ad_tag (tokens):
	temp = []
	for i in range (0, len (tokens)):
		if tokens[i] == "@": continue				# Do not copy the @
		if i > 0 and tokens[i-1] == "@": continue	# Do not copy anything right after @
		temp.append (tokens[i])
	return temp

# Remove all # tag in the token list
def remove_hash_tag (tokens):
	temp = []
	for i in range (0, len (tokens)):
		if tokens[i] == "#": continue				# Do not copy the #
		if i > 0 and tokens[i-1] == "#": continue	# Do not copy anything right after #
		temp.append (tokens[i])
	return temp

# Remove all stopwords in the token list
def remove_stopwords (tokens, stopword_list):
	return [token for token in tokens if token not in stopword_list]

# Get rid of all non-alphabet token
def remove_nonalphabet (tokens):
	return [token for token in tokens if token.isalpha ()]

# Stem the token into the normalized from
def stem_tokens (tokens):
	stemmer = PorterStemmer ()
	return [str (stemmer.stem_word (token)) for token in tokens]	# str () is to convert the possible unicode string into normal string

# Concatinate all tokens into a single string
def join_tokens (tokens):
	content = ""
	for token in tokens:
		content = content + " " + token
	return content

# Process the raw data from user
def process_data (train_data):
	# Remove all irrelevant record
	train_data = [row for row in train_data if row[1] != "irrelevant"]

	temp = []
	# Tokenize, stem, and remove all unnecessary information 
	for row in train_data:
		content = row[0]
		sentiment = row[1]

		content = remove_url (content)
		tokens = tokenize (content)
		tokens = remove_ad_tag (tokens)
		tokens = remove_hash_tag (tokens)
		tokens = remove_nonalphabet (tokens)
		tokens = stem_tokens (tokens)
		tokens = remove_stopwords (tokens, stopword_list.get_stopwords())
		content = join_tokens (tokens)

		# Skip the row if the content is shorter than 3 characters
		if (len (content) > 3):
			temp.append ((content, sentiment))

	return temp

# Return the classifier that is trained by the data provided
def get_trained_classifier (train_data):
	classifier = NaiveBayesClassifier (train_data)
	return classifier






#### Main Program
train_data = load_database (params.TRAINER_PARAM_INPUT_FILE_NAME, params.TRAINER_PARAM_TRAIN_SIZE)
train_data = process_data (train_data)
classifier = get_trained_classifier (train_data)


