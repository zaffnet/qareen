import unittest
from pathlib import Path

from qareen.config.settings import Settings


class TestSettings(unittest.TestCase):

    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.default_embedding_models, ["google/siglip-base-patch16-224"])
        self.assertEqual(settings.data_dir, Path("data/"))
        self.assertEqual(settings.chroma_db_dir, Path("chroma_db/"))
        self.assertEqual(settings.dev_sample_size, 1000)
        self.assertEqual(settings.environment, "dev")
        self.assertEqual(settings.alphas, [0.5])

if __name__ == '__main__':
    unittest.main()
