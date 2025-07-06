import logging
import os
os.environ["MODIN_ENGINE"] = "ray"
import matplotlib.pyplot as plt
from utils.utils import DataLoader, TextPreprocessor, DatasetManager
from utils.simple_utils import KeywordBasedClassifier
import time
import warnings

import argparse

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.simplefilter('ignore')




def main(args) -> None:

    # Use args to control dataset, log file, minimum_count_samples, and models
    logging.info("\n\n\nStarting main function with arguments: %s", args)
    print("Main function started with arguments. Check logs for details.")

    # Load the dataset from the specified directory
    if not os.path.exists(args.dataset):
        logging.error("Dataset directory not found: %s", args.dataset)
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")
    else:
        logging.info("Dataset directory found: %s", args.dataset)
        print(f"Dataset directory found: {args.dataset}")
        preprocessor = TextPreprocessor()
        manager = DatasetManager(logging, preprocessor)
        dataset_name = args.dataset
        df = DataLoader.load_data(dataset_name)

        args.manager = manager
        args.preprocessor = preprocessor

        clf = KeywordBasedClassifier(args, logging)
        logging.info("KeywordBasedClassifier initialized with args: %s", args)

    print(df.head())
    print(df['label'].value_counts())
    logging.info("Data loaded into DataFrame:\n%s", df['label'].value_counts())


    #############################################################
    ##      Orginal Data Processing
    #############################################################
    # Original data processing is kept as is
    if args.minimum_count_samples is None or "org" in args.minimum_count_samples:
        logging.info("---------- Test On Orginal Data\n")
        try:
            print("Starting original data processing...")
            logging.info("Starting original data processing...\n")
            # Call the classification function with the original data
            logging.info("Calling classification_basic function with original data...\n")
            print("Calling classification_basic function with original data...")
            # The classification_basic function will perform keyword-based categorization and TF-IDF + classifier
            # classification
            df_corrected, df_train, df_test = manager.train_test_split_and_process(df, test_size=0.3, preprocess=True, random_state=42)
            logging.info("Data split into training and testing sets for original data.\n")
            print("Data split into training and testing sets for original data.")
            clf.simple_classification(df_train, df_test, log_name='original_data')

            print("Original data processing completed successfully.")
        except ValueError as ve:
            logging.error("ValueError during original data processing: %s", str(ve), exc_info=True)
            print(f"A ValueError occurred during original data processing: {ve}")
        except Exception as e:
            logging.error("Error during original data processing: %s", str(e), exc_info=True)
            print(f"An error occurred during original data processing: {e}")
        logging.info("---------- Test On Orginal Data\n")
    else:
        logging.info("Skipping original data processing as per args.minimum_count_samples setting.\n")
        print("Skipping original data processing as per args.minimum_count_samples setting.\n")

    #############################################################
    ##      Balanced Data Processing (based on args)
    #############################################################
    if args.minimum_count_samples is not None:
        clf.process_min_count_simple(df)
          



if __name__ == "__main__":
    """Main entry point for the simple methods of long document classification script.
    
    This script performs simple methods of long document classification on arXiv data using various methods and parameters.
    It supports both original data processing and balanced data processing with configurable minimum 
    sample counts per class.
    
    Command line arguments:
        --dataset: Path to the dataset directory (default: 'dataset/Long-document-dataset')
        --log: Path to the log file (default: 'log/simple_methods.log') 
        --minimum_count_samples: List of minimum counts for class balancing (default: ["org", "minlength", "100", "140"])
        --numkey: Number of keywords to extract (default: 50)
        --method: Method for keyword extraction - "count" or "tfidf" (default: "tfidf")
        
    The script processes the data based on the provided arguments, performs classification using 
    specified methods, and logs results and performance metrics to the specified log file.
    """

    parser = argparse.ArgumentParser(description="simple methods of long document classification on arXiv data.")
    parser.add_argument('--dataset', type=str, default='dataset/Long-document-dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--log', type=str, default='log/simple_methods.log',
                        help='Path to the log file')
    parser.add_argument('--minimum_count_samples', type=str, nargs='+', default=["org", "minlength", "100", "140"],
                        metavar='N',
                        help='List of minimum counts of samples per class for balancing. Use "minlength" to automatically use the minimum class size. Use "org" to automatically use the orginal class size.')
    parser.add_argument('--numkey', type=int, default=512,
                        help='Number of keywords to extract')
    parser.add_argument('--method', type=str, choices=['count', 'tfidf'], default='tfidf',
                        help='Method for keyword extraction: "count" for simple word count, "tfidf" for TF-IDF based keyword extraction')
    args = parser.parse_args()

    # Update dataset path and log file if provided
    if args.dataset:
        args.dataset = os.path.abspath(args.dataset)
    if args.log:
        args.log = os.path.abspath(args.log)
    # Set up logging
    if not os.path.exists(os.path.dirname(args.log)):
        os.makedirs(os.path.dirname(args.log))

    logging.basicConfig(filename=args.log, level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    start_script_time = time.time()
    # Optionally, you can refactor main() to accept args if needed
    main(args)
    end_script_time = time.time()
    total_running_time = end_script_time - start_script_time
    logging.info(f"Total script running time: {total_running_time:.2f} seconds")
    print(f"Total script running time: {total_running_time:.2f} seconds")
    logging.info("Main function completed successfully.")
    print("Main function completed successfully. Check logs for details.")