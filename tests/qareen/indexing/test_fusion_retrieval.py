import unittest
from unittest.mock import MagicMock, patch
from qareen.indexing.chroma_indexer import ChromaIndexer, Chroma
from langchain_core.documents import Document
from qareen.config.settings import Settings
from qareen.indexing.models import EmbeddingModel


class TestFusionRetrieval(unittest.TestCase):
    def setUp(self):
        self.dataset_loader = MagicMock()
        self.dataset_loader.get_dataset_name.return_value = "test_dataset"

        self.embedding_model = MagicMock(spec=EmbeddingModel)
        self.embedding_model.get_model_id.return_value = "test_model"

        self.settings = Settings(chroma_db_dir="tmp", environment="dev")
        self.indexer = ChromaIndexer(self.dataset_loader, self.embedding_model, self.settings)

    def test_query_rrf_combines_scores(self):
        # Mock _query_linear to return different results for alpha=0 and alpha=1

        # Text results (alpha=0)
        doc1 = Document(page_content="doc1", metadata={"_id": "1"})
        doc2 = Document(page_content="doc2", metadata={"_id": "2"})
        # Image results (alpha=1) - doc2 is better here, doc3 is new
        doc3 = Document(page_content="doc3", metadata={"_id": "3"})

        # Side effect for _query_linear to return results based on alpha
        def mock_query_linear(**kwargs):
            alpha = kwargs.get("alpha")
            # Return list of (doc, score)
            if alpha == 0.0:
                return [(doc1, 0.9), (doc2, 0.8)]  # doc1 rank 1, doc2 rank 2
            elif alpha == 1.0:
                return [(doc2, 0.9), (doc3, 0.8)]  # doc2 rank 1, doc3 rank 2
            return []

        with (
            patch.object(self.indexer, "get_collection_name", return_value="mock_collection"),
            patch.object(self.indexer, "create_vectorstore"),
            patch.object(self.indexer, "_query_linear", side_effect=mock_query_linear),
        ):
            # alpha=0.5 (equal weight)
            results = self.indexer.query_multimodal(
                vectorstore=MagicMock(),
                image=None,
                text="query",
                alpha=0.5,
                k=3,
                combination_method="rrf",
            )

            # Expected RRF calculation (k=60)
            # doc1 (rank 1 in text): 0.5 * (1/61) = 0.0081967
            # doc2 (rank 2 in text, rank 1 in image):
            #   0.5 * (1/62) + 0.5 * (1/61) = 0.0080645 + 0.0081967 = 0.0162612
            # doc3 (rank 2 in image): 0.5 * (1/62) = 0.0080645

            # Order should be doc2, doc1, doc3
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0][0].page_content, "doc2")
            self.assertEqual(results[1][0].page_content, "doc1")
            self.assertEqual(results[2][0].page_content, "doc3")

            rrf_k = 60
            score_doc2 = 0.5 * (1 / (rrf_k + 1)) + 0.5 * (1 / (rrf_k + 2))
            self.assertAlmostEqual(results[0][1], score_doc2, places=5)

    def test_mmr_mode_calls_mmr_search(self):
        # Mock Chroma vectorstore
        mock_vectorstore = MagicMock(spec=Chroma)

        mock_vectorstore.max_marginal_relevance_search_by_vector.return_value = [
            Document(page_content="mmr_doc", metadata={})
        ]

        # Mock embedding
        self.embedding_model.embed_multimodal.return_value = MagicMock(tolist=lambda: [0.1, 0.2])

        results = self.indexer._query_linear(
            vectorstore=mock_vectorstore,
            image=None,
            text="query",
            alpha=0.5,
            k=1,
            score_threshold=None,
            retrieval_mode="mmr",
            distance_metric="cosine",
        )

        mock_vectorstore.max_marginal_relevance_search_by_vector.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page_content, "mmr_doc")
        self.assertEqual(results[0][1], 0.0)  # Expect placeholder score

    def test_l2_distance_metric_passed(self):
        # Verify distance_metric arg is passed down to create_vectorstore in RRF
        with (
            patch.object(self.indexer, "_query_linear"),
            patch.object(self.indexer, "create_vectorstore") as mock_create,
            patch.object(self.indexer, "get_collection_name", return_value="mock_collection"),
        ):
            self.indexer.query_multimodal(
                vectorstore=MagicMock(),
                image=None,
                text="query",
                alpha=0.5,
                k=1,
                combination_method="rrf",
                distance_metric="l2",
            )

            # check create_vectorstore calls
            # It should be called with distance_metric="l2"
            calls = mock_create.call_args_list
            # Check keyword args first
            found = False
            for call in calls:
                if call.kwargs.get("distance_metric") == "l2":
                    found = True
                    break
                # If positional, it's the 5th arg (index 4)
                if len(call.args) > 4 and call.args[4] == "l2":
                    found = True
                    break

            self.assertTrue(found, "create_vectorstore was not called with distance_metric='l2'")


if __name__ == "__main__":
    unittest.main()
