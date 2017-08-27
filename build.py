from collections import defaultdict
from gensim.models import doc2vec
from gensim import matutils
import logging
import cPickle as pickle
import cProfile

# Load Tagged Documents from pickle
with open('pickle/TaggedDocuments', 'r') as f:
	documents = pickle.load(f)      # type: list of doc2vec.TaggedDocument

# Load Sentences corpus from pickle
with open('pickle/Sentences', 'r') as f:
	sentences = pickle.load(f)      # type: list of string

class Model:
	def __init__(self, documents):
		self.documents = documents
		self.model = None       # :type do2vec.Doc2Vec
		
	def config(self):
		'''
		Configure trainedmodel setting
		Load pretrained word vectors from GoogleNew corpus
		:rtype: doc2vec.Doc2vec
		'''
		default_alpha = 0.025
		min_alpha = 0.0001
		windows = 8;
		
		model = doc2vec.Doc2Vec(size=300, min_count=1, window= windows, alpha= default_alpha, min_alpha= min_alpha)
		model.build_vocab(documents)
		model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, lockf = 0.0)
		self.model = model
	
	def dump_pickle(self):
		with open('pickle/Model', 'w') as f:
			pickle.dump(self, f)
	
	def train_epoch(self, epoch):
		"""
		Return a trained trainedmodel based on base trainedmodel & epoch time
		:param epoch:
		:return:
		"""
		model = self.model
		assert isinstance(model, doc2vec.Doc2Vec)
		
		total_examples = len(self.documents)
		for iter in range(epoch):
			model.train(documents, total_examples=total_examples, epochs=1)
		return model
			

def record_model(model, epoch):
	m = model.train_epoch(epoch)
	assert isinstance(m, doc2vec.Doc2Vec)
	with open('trainedmodel/epoch' + str(epoch), 'w') as f:
		m.save(f)

def test():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# m = Model(documents)
	# m.config()
	# m.dump_pickle()
	
	# load trainedmodel
	with open('pickle/Model', 'r') as f:
		m = pickle.load(f)      # type: Model
		assert isinstance(m.model, doc2vec.Doc2Vec)
	
	record_model(m, 3)
	
	m = doc2vec.Doc2Vec.load('trainedmodel/epoch3')
	assert isinstance(m, doc2vec.Doc2Vec)
	
	print len(m.docvecs)
	
# cProfile.run('test()')
test()