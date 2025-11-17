import argparse
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader

def main():
    parser = argparse.ArgumentParser(description="Download the SQID dataset from HuggingFace.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="zaffnet/SQID",
        help="The name of the dataset to download from HuggingFace.",
    )
    args = parser.parse_args()

    print(f"Downloading {args.dataset_name}...")
    loader = HuggingFaceDatasetLoader(dataset_name=args.dataset_name)

    try:
        dataset = loader.load()
        print(f"Dataset '{dataset.dataset_name}' downloaded successfully.")
        print(f"Number of items: {len(dataset.data)}")
    except Exception as e:
        import sys
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
