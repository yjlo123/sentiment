from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import csv
import sys
 
train = []

f = open('train.csv', 'rt')
reader = csv.reader(f)
next(reader); #skip header

for n in range(0,100):
    row = next(reader)
    content = row[3].strip()
    sentiment = row[4].strip()[:3]
    if sentiment == "pos" or sentiment == "neg":
        train.append((content, sentiment))

#print train

print "Training..."
cl = NaiveBayesClassifier(train)
print "Finished."

# Classify some text
print(cl.classify("50% of FoxNews.com readers think #hcr won't pass. The other 50% can't get Internet service at the trailer park."))  # "neg"
print(cl.classify("There is nothing more #pro-life than protecting the lives of 31 million Americans.'' http://bit.ly/b8DXel #hcr #passthedamnbill"))  # "pos"



'''
# Classify a TextBlob
blob = TextBlob("The beer was amazing. But the hangover was horrible. "
                "My boss was not pleased.", classifier=cl)
print(blob)
print(blob.classify())
 
for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())
 
# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))
 
# Show 5 most informative features
cl.show_informative_features(5)
'''