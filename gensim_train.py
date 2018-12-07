import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import re
import numpy as np
import csv


# nltk.download('stopwords')
stop = set(stopwords.words('english'))
# nltk.download('wordnet')
lemma = WordNetLemmatizer()

class Posts(object):
	def __init__(self, filename):
		self.file = filename

	def __iter__(self):
		with open(self.file, 'r') as f:
			csv_reader = csv.reader(f, delimiter=',')
			next(csv_reader)
			cnt = 0
			for row in csv_reader:
				if cnt % 10000 == 0:
					print("Done with %d posts..." %(cnt))
				cnt += 1

				post = row[1]
				post = post.lower()
				body = re.findall(r"[A-Za-z]+|\S", post)
				stop_free = " ".join([i for i in body if i not in stop])
				normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
				normalized = re.sub(r"\b\d+\b", "NUM", normalized)
				normalized = re.sub(r"\b[a-zA-Z]\b", "CHAR", normalized)

				yield normalized.split()
				

filename = 'baseline-data/baseline_train_data.csv'
posts = Posts(filename)
model = gensim.models.Word2Vec(posts, min_count=10, size=300, iter=5, workers=4)
model.save('./gensim_wordmodel')

# print(model['namespace'])
# print(model['#'])


