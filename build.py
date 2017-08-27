from gensim.models import doc2vec
import cPickle as pickle

# Load Tagged Documents from pickle
with open('pickle/TaggedDocuments', 'r') as f:
	documents = pickle.load(f)      #type: doc2vec.TaggedDocument
	
class Model:
	"""
	:type documents: list of doc2vec.TaggedDoucment
	"""
	def __init__(self, documents):
		self.documents = documents
		self.model = None
	
	def config(self):
		'''
		Configure model setting
		Load pretrained word vectors from GoogleNew corpus
		:rtype: doc2vec.Doc2vec
		'''
		default_alpha = 0.025
		min_alpha = 0.0001
		windows = 8;
		
		model = doc2vec.Doc2Vec(size=300, min_count=1, workers=4, window= windows, alpha= default_alpha, min_alpha= min_alpha)
		model.build_vocab(documents)
		model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
		self.model = model
			
	def dump_pickle(self):
		with open('pickle/Model', 'w') as f:
			pickle.dump(self, f)
	
def main():
	
	m = Model(documents)
	m.dump_pickle()
	
	# load model
	with open()