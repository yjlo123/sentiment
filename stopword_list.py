from nltk.corpus import stopwords

def get_stopwords ():
	custom_stopwords = ["RT"]
	stopword_list = stopwords.words('english') + custom_stopwords

	return stopword_list

