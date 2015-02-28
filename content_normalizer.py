from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import PunktWordTokenizer
from nltk.stem import PorterStemmer
import stopword_list
import re


class ContentNormalizer:
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
		content = ContentNormalizer.__remove_url (content)
		tokens = ContentNormalizer.__tokenize (content)
		tokens = ContentNormalizer.__remove_ad_tag (tokens)
		tokens = ContentNormalizer.__remove_hash_tag (tokens)
		tokens = ContentNormalizer.__remove_nonalphabet (tokens)
		tokens = ContentNormalizer.__stem_tokens (tokens)
		tokens = ContentNormalizer.__remove_stopwords (tokens, stopword_list.get_stopwords())
		content = ContentNormalizer.__join_tokens (tokens)
		content = content.strip ()

		return content