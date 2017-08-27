from gensim.models import doc2vec
import cPickle as pickle

# Load Tagged Documents from pickle
with open('pickle/TaggedDocuments', 'r') as f:
	documents = pickle.load(f)      # type: doc2vec.TaggedDocument


def test():
	print len(documents)
test()