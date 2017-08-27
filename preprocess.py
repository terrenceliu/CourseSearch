from collections import defaultdict
import pandas as pd
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')

class CourseCatalog(object):
	def __init__(self, f):
		"""
		
		:param f: excel file of course catalog
		:type f: file
		:type self.catalog: pd.Dataframe
		"""
		
		self.df = pd.read_excel(f)      # type: pd.Dataframe
		self.training_set = []
		
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
	
	def stem_words(sel):
		"""
		Stemming every word into unified form
		:return:
		"""
	
def test():
	with open('data/catalog.xlsx', 'r') as f:
		Catalog = CourseCatalog(f)
	assert isinstance(Catalog.df, pd.DataFrame)
	
	Catalog.fill_na()
	Catalog.tag_sentences()
	Catalog.tokenize_sentence()
	# Catalog.remove_stop_words(STOP_WORDS)
	print Catalog.training_set[-200:-180]
	
	
	
test()