from .abstract_embedding import AbstractEmbedding
from gensim.models import Word2Vec
import os
from abc import  abstractmethod
import numpy as np

class RandomWalkEmbedding(AbstractEmbedding):
    def __init__(self, d, network):
        super().__init__(d,network)        
        self.num_walks = None
        self.walk_length = None 
        self.window = None   
        #
        self.walks = None
        self.model = None

    @abstractmethod
    def _simulate_walks(self):
        pass

    def compute_embedding(self, num_walks = 80, walk_length = 40, window = 10):
        self.num_walks = num_walks
        self.walk_length = walk_length 
        self.window = window        
        #if self.walks is None:
        self._simulate_walks()
        
        # just need to run random walks an then pass them to gensim
        self.model = Word2Vec(self.walks, #.walks, 
                         size=self.d, 
                         window= self.window, 
                         min_count=0, # removes words appeared less than
                         sg = 1, # ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW
                         workers= os.cpu_count(), 
                         batch_words = 1000 #Target size (in words) for batches of examples passed to worker threads, default is 10000
                        )
        nodes_in_vocabulary = set([tuple(k.split("-")) for k in self.model.wv.vocab.keys()])
        self.embedding = {
            tuple(k.split("-")):
            self.model.wv.word_vec(k) for k in self.model.wv.vocab.keys()}

        #add coordinates for nodes that were not observed through rw
        for node in set(self.nodes).difference(nodes_in_vocabulary): #network.nodes.index.keys()
            print("Node \"{}\" is in no path and has been given random coordinates ".format(node))
            self.embedding[node] = np.random.rand(self.d)