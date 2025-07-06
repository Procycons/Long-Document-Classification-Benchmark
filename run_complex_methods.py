import argparse
from utils.complex_utils import TransformerTextClassifier
MAX_LEN = 512  # Default maximum sequence length for transformer models


if __name__ == "__main__":
    """
    CLI for training or testing a transformer model using the TransformerTextClassifier class.
    """

    default_args = {
        'dataset_name': 'dataset/Long-document-dataset',
        'min_count_samples': 100,
        'epochs': 100,
        'train_bachSize': 32,
        'eval_bachSize': 8,
        'model_name': "roberta-base", # bert-base-uncased, distilbert-base-uncased, roberta-base
        'max_len': MAX_LEN,
        'mode': 'train',  # 'train' or 'test'
        'test_dataset_name': None,
        'model_dir': None,
        'log': None,  # Path to save logs
    }

    parser = argparse.ArgumentParser(
        description="Train or evaluate a transformer model for long document classification.")

    for arg, val in default_args.items():
        arg_type = type(val) if val is not None else str
        help_text = f'Default: {val}'
        parser.add_argument(f'--{arg}', type=arg_type, default=val, help=help_text)

    args = parser.parse_args()

    # Ensure the dataset directory exists
    if not args.dataset_name or not args.dataset_name.strip():
        raise ValueError("Dataset name must be provided and cannot be empty.")
    if not args.model_name or not args.model_name.strip():
        raise ValueError("Model name must be provided and cannot be empty.")
    if args.test_dataset_name is None:
        args.test_dataset_name = args.dataset_name

    # Initialize the classifier
    classifier = TransformerTextClassifier(args)

    if args.mode == 'train':
        classifier.train()
    elif args.mode == 'test':
        classifier.test()
    else:
        print("Invalid mode. Use --mode train or --mode test.")