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

# Define the record tuple
Record = collections.namedtuple ("Record", "content sentiment")

class Tester:
	def __init__ (self):
		self.__test_data = []
		self.__feature_extractor = FeatureExtractor ()

	# Read the database file into the train_data field as a list of Record tuple
	def __load_database (self, filename, test_size):
		# Load the test database into memory
		self.__test_data = []
		f_input = open (filename, 'rt')
		reader = csv.reader (f_input)
		next (reader)

		record_read = 0
		for row in reader:
			# Extract the content and setiment
			content = TextBlob (row[3].strip ())
			sentiment = TextBlob (row[4].strip ())

			# Append to the train_data
			self.__test_data.append (Record (content, sentiment))

			# Count the record read, stop if the desired has been reached
			record_read = record_read + 1
			if test_size >= 0 and record_read >= test_size:
				break
		
	# Train the classifier using the database provided
	def test_classifier (self, trainer, filename, test_size):
		self.__load_database (filename, test_size)
		# Process the data
		self.__process_data (trainer)
	
	# Process the raw data from user
	def __process_data (self, trainer):

		temp = []
		match_count = 0
		dev_cat_count = 0
		result_cat_count = 0
		eval_dic = {
			"positive":[0,0,0],
			"negative":[0,0,0],
			"neutral":[0,0,0],
			"irrelevant":[0,0,0]
		}
		#print "Evaluating..."
		# Tokenize, stem, and remove all unnecessary information 
		for row in self.__test_data:
			content = ContentNormalizer.normalize_content (row.content)
			sentiment = row.sentiment
			classify_result = trainer.classify(content)
			#print str(classify_result)+"=="+str(sentiment)
			test_result = classify_result

			if test_result[0] != "irrelevant":
				test_result = str(test_result[0][0])
			else:
				test_result = str(test_result[0])

			#print test_result+" -> "+str(sentiment)
			for eval_cat in eval_dic:
				if test_result == eval_cat and str(sentiment) == eval_cat:
					eval_dic[eval_cat][0] += 1
				if test_result == eval_cat:
					eval_dic[eval_cat][1] += 1
				if str(sentiment) == eval_cat:
					eval_dic[eval_cat][2] += 1
		#print self.__calc_f1(match_count, result_cat_count, dev_cat_count)
		#self.__train_data = temp
		f1 = 0
		for cat in eval_dic:
			f1 += self.__calc_f1(eval_dic[cat][0], eval_dic[cat][1], eval_dic[cat][2])
		f1 = f1 / len(eval_dic)
		print f1
		print eval_dic
		'''
		for cat in eval_dic:
			print float(eval_dic[cat][0])/eval_dic[cat][2], #match/should(dev_count)
		'''

	# Calculate F1
	def __calc_f1 (self, match_count, result_cat_count, dev_cat_count): 
		a = match_count			# relevant & retrieved
		b = result_cat_count	# retrieved
		c = dev_cat_count		# relevant

		if b == 0:
			p = 0
		else:
			p = float(a)/b
		if b == 0:
			r = 0
		else:
			r = float(a)/c
		f1 = 2*p*r/(p+r+1) #plus 1 smoothing
		return f1

# Get train data and train the classifier
trainer = trainer.Trainer ()
trainer.train_classifier (params.TRAINER_PARAM_INPUT_FILE_NAME, params.TRAINER_PARAM_TRAIN_SIZE)
trainer.train_analyzer (params.FEATURE_FILE_NAME)

tester = Tester()
tester.test_classifier(trainer, params.DEV_PARAM_INPUT_FILE_NAME, params.DEV_PARAM_TRAIN_SIZE)

for n in range(0, 9):
	#print str(params.TRAINER_PARAM_TRAIN_SIZE)
	'''
	trainer.train_classifier (params.TRAINER_PARAM_INPUT_FILE_NAME, params.TRAINER_PARAM_TRAIN_SIZE)
	trainer.train_analyzer (params.FEATURE_FILE_NAME)

	print ""
	print str(params.STRENGTH_THRESHOLD[0])
	tester.test_classifier(trainer, params.DEV_PARAM_INPUT_FILE_NAME, params.DEV_PARAM_TRAIN_SIZE)
	params.STRENGTH_THRESHOLD[0] += 0.05
	'''