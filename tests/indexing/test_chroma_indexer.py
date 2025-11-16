import unittest
from qareen.indexing.chroma_indexer import ChromaIndexer

class TestChromaIndexer(unittest.TestCase):

    def test_get_collection_name(self):
        indexer = ChromaIndexer()
        collection_name = indexer.get_collection_name(
            dataset_name="sqid",
            model_id="google/siglip-base-patch16-224",
            alpha=0.5,
            environment="dev"
        )
        self.assertEqual(collection_name, "dev_sqid_google-siglip-base-patch16-224_0.5")

    def test_index(self):
        indexer = ChromaIndexer()
        with self.assertRaises(NotImplementedError):
            indexer.index(data=None, model=None, alpha=0.5)

if __name__ == '__main__':
    unittest.main()
