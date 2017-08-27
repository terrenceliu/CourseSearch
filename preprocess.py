from collections import defaultdict
import pandas as pd
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from gensim.models import doc2vec
import cPickle as pickle

STOP_WORDS = stopwords.words('english')

class CourseCatalog(object):
	def __init__(self, f):
		"""
		
		:param f: excel file of course catalog
		:type f: file
		:type self.catalog: pd.Dataframe
		"""
		
		self.df = pd.read_excel(f)      # type: pd.Dataframe
		self.training_set = []          # type: list
		
	def fill_na(self):
		self.df = self.df.fillna('')
		
	def to_excel(self, filename):
		"""
		Save catalog into excel formate file
		
		:param filename:
		:return:
		"""
		writer = pd.ExcelWriter(filename)
		self.df.to_excel(writer, 'Sheet1')
		writer.save()
			
	def tag_sentences(self):
		"""
		Tokenize content into sentences.
		Tag sentences in df content and save to training set
		Format: ([sentence], tag)
		:return:
		"""
		result = []
		for index, row in self.df.iterrows():
			for content in row:
				sentences = sent_tokenize(content)
				for e in sentences:
					result.append((e, index))
		self.training_set = result
		
	def tokenize_sentence(self):
		"""
		Tokenize the sentence into a list of words.
		:return:
		"""
		
		result = []
		for ele in self.training_set:
			tag = ele[1]
			# Use word_tokenize; Should it use wordpunct_tokenize?
			words = word_tokenize(ele[0])
			result.append((words, tag))
		self.training_set = result
	
	def expand_vocab(self):
		"""
		For each word in corpus, expand top-n related words in Google News Corpus.
		:return:
		"""
		
	def remove_stop_words(self, stop_words):
		"""
		Return a training set that is removed of given stop words
		:param stop_words:
		:return:
		"""
		result = []
		for ele in self.training_set:
			tag = ele[1]
			sent = [word for word in ele[0] if word not in stop_words]
			result.append((sent, tag))
		self.training_set = result
	
	def stem_words(self):
		"""
		Stemming every word into unified form
		:return:
		"""
	
	def course_code(self):
		"""
		Return a list of course code
		:return:
		"""
		return self.df.index.values
	
	def dump_pickle(self):
		with open('pickle/Catalog', 'w') as f:
			pickle.dump(self, f)


def create_tagged_document(catalog):
	"""
	Create tagged document for Doc2Vec model from Course Catalog
	:param catalog:
	:type catalog: CourseCatalog
	:return:
	"""
	documents = []
	for ele in catalog.training_set:
		sent = ele[0]
		tag = ele[1]
		instance = doc2vec.TaggedDocument(sent, tag)
		documents.append(instance)
	return documents

def main():
	# Init CourseCatalog
	with open('data/catalog.xlsx', 'r') as f:
		Catalog = CourseCatalog(f)
	
	assert isinstance(Catalog.df, pd.DataFrame)
	# Tweak
	Catalog.fill_na()
	Catalog.tag_sentences()
	Catalog.tokenize_sentence()
	# Catalog.remove_stop_words(STOP_WORDS)
	Catalog.dump_pickle()
	
	# Create Tagged Document for Doc2Vec Model
	documents = create_tagged_document(Catalog)
	with open('pickle/TaggedDocuments', 'w') as f:
		pickle.dump(documents, f)
	
def test():
	# Load from pickle
	with open('pickle/Catalog', 'r') as f:
		Catalog = pickle.load(f)        # type: CourseCatalog

main()