import unittest
from qareen.indexing.models import EmbeddingModel
from qareen.indexing.base import VectorStoreIndexer

class TestAbstractIndexingClasses(unittest.TestCase):

    def test_cannot_instantiate_embedding_model(self):
        with self.assertRaises(TypeError):
            EmbeddingModel()

    def test_cannot_instantiate_vector_store_indexer(self):
        with self.assertRaises(TypeError):
            VectorStoreIndexer()

if __name__ == '__main__':
    unittest.main()
