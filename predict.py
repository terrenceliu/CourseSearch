from collections import defaultdict
from gensim.models import doc2vec
from gensim import matutils
from numpy import dot, array
from numpy.linalg import norm
import logging
import cPickle as pickle
import cProfile

FILE_PATH = 'trainedmodel/'

def load_model(filename):
	"""
	Load trained Doc2Vec model from 'path/to/name'
	:param filename:
	:return: mdoel
	:rtype: doc2vec.Doc2Vec
	"""
	model = doc2vec.Doc2Vec.load(filename)
	return model


def predict_course(model, query_list, n = 5):
	"""
	Given a pretrianed model and a list of query words, return the top-n related course with cosine similarity
	:param n: get top-n related course; default n = 5
	:param model: a doc2vec model
	:param query_list: a list of query word
	:type model: doc2vec.Doc2Vec
	:type query_list: list
	:return: [(tag, cosine_similarity)]
	"""
	dv = model.docvecs
	
	def get_vector(model, query_list):
		"""
		Return the sum vector of a list of words
		:type model: doc2vec.Doc2Vec
		:type query_list: list
		"""
		strlist = query_list.split(" ")
		v = [model[i] for i in strlist]
		return matutils.unitvec(array(v).sum(axis=0))
	
	sum_vec = get_vector(model, query_list)
	result = model.docvecs.most_similar([sum_vec], topn = n)
	return result

def test():
	"""
	Testing
	:return:
	"""
	ep3 = FILE_PATH + 'epoch3'
	model = load_model(ep3)
	
	query_list = 'computer science'
	print predict_course(model, query_list)
test()