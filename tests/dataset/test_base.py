import unittest
from qareen.dataset.base import DatasetLoader

class TestAbstractDatasetLoader(unittest.TestCase):

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            DatasetLoader()

    def test_can_subclass(self):
        class ConcreteLoader(DatasetLoader):
            def load(self):
                pass
            def validate_schema(self, data):
                pass
            def get_dataset_name(self):
                pass
            def get_dataset_info(self):
                pass

        try:
            ConcreteLoader()
        except TypeError:
            self.fail("Should be able to instantiate a concrete subclass of DatasetLoader")

if __name__ == '__main__':
    unittest.main()
