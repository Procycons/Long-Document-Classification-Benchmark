import logging
import argparse
import os
os.environ["MODIN_ENGINE"] = "ray"
import time
import warnings
warnings.simplefilter('ignore')


from utils.utils import DataLoader, TextPreprocessor, DatasetManager
from utils.intermediate_utils import IntermediateClassifier



def main(args) -> None:
    """Main function for running Intermediate long document classification experiments.

    Processes long document classification data and runs experiments using specified models.
    Supports both original unbalanced data and balanced datasets with configurable 
    minimum samples per class.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            dataset (str): Path to dataset directory
            log (str): Path to log file
            minimum_count_samples (List[str]): List of minimum sample counts for balancing
                "minlength" - Use minimum class size
                "org" - Use original unbalanced data
                numeric values - Specific minimum counts
            models (List[str]): Models to use (reg, svm, mlp, xgb)

    The function:
    1. Loads and preprocesses the dataset
    2. Runs classification on original unbalanced data if specified
    3. Creates balanced datasets with specified minimum counts
    4. Runs classification experiments on balanced datasets
    5. Logs results, metrics and visualizations

    Raises:
        ValueError: If DataFrame lacks required 'label' column for balancing
        Exception: For other processing errors during classification

    Results are logged to the specified log file including:
    - Data statistics and shapes
    - Model performance metrics
    - Confusion matrices 
    - Execution times
    """
    
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
        df = DataLoader.load_data(args.dataset)

        clf = IntermediateClassifier(args=args, logger=logging, preprocessor=preprocessor, manager=manager)
    print(df.head())
    print(df['label'].value_counts())
    logging.info("Data loaded into DataFrame:\n%s", df['label'].value_counts())

    

    #############################################################
    ##      Orginal Data Processing
    #############################################################
    if args.minimum_count_samples is None or "org" in args.minimum_count_samples:
        logging.info("---------- Test On Orginal Data\n")
        try:
            print("Starting original data processing...")
            logging.info("Starting original data processing...\n")

            df_corrected, df_train, df_test = manager.train_test_split_and_process(df, test_size=0.3, preprocess=True, random_state=42)
            print("Shapes of training and testing sets:")
            logging.info("Shapes of training and testing sets:\nX_train: %s, y_train: %s, X_test: %s, y_test: %s",
                                df_train.shape, df_train['label'].shape, df_test.shape, df_test['label'].shape)
            print(f"X_train shape: {df_train.shape}, y_train shape: {df_train['label'].shape}, X_test shape: {df_test.shape}, y_test shape: {df_test['label'].shape}")
            logging.info("Applied cleaning functions directly to the Series X_train and X_test\n")

            # Call the classification function with the original data
            logging.info("Calling classification_basic function with original data...\n")
            print("Calling classification_basic function with original data...")
            # The classification_basic function will perform keyword-based categorization and TF-IDF + classifier
            # classification
            clf.classify(df_train["text"], df_train["label"], df_test["text"], df_test["label"], "org_dataset")
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
        min_counts = []
        for val in args.minimum_count_samples:
            if isinstance(val, str) and val.lower() == "org":
                continue  # Skip original data processing
            elif isinstance(val, str) and val.lower() == "minlength":
                # Use the minimum count of samples in any class
                min_counts.append(df['label'].value_counts().min())
            else:
                min_counts.append(int(val))
        # Using joblib to parallelize the loop over different min_counts
        logging.info(f"Starting  processing for minimum counts: {min_counts}\n")
        print(f"Starting  processing for minimum counts: {min_counts}")
        for min_count in min_counts:
            clf.process_with_min_count(df, min_count)
            logging.info("End of processing for min_count set on %s\n", min_count)
            print("processing for minimum counts completed.")
        
    logging.info("Main function completed successfully.")
    print("Main function completed successfully. Check logs for details.")

if __name__ == "__main__":

    """Main entry point for the arXiv text classification script with intermediate methods.
    
    This script performs intermediate text classification on arXiv paper data using multiple models:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Multi-layer Perceptron (MLP) 
    - XGBoost

    The script supports:
    - Processing original unbalanced data
    - Creating balanced datasets with configurable minimum samples per class
    - Running classification experiments on both original and balanced data
    - Logging results, metrics and visualizations

    Command line arguments:
        --dataset: Path to dataset directory containing labeled text files
        --log: Path to output log file
        --minimum_count_samples: List of minimum sample counts for class balancing
                               Use "minlength" for automatic minimum class size
                               Use "org" for original unbalanced data
        --models: List of models to use (reg, svm, mlp, xgb)

    Directory structure:
        dataset/Long-document-dataset/
        ├── label1/
        │   ├── file1.txt
        ...

    Results are logged including:
    - Data statistics
    - Model performance metrics 
    - Confusion matrices
    - Execution times
    - Memory usage
    """


    parser = argparse.ArgumentParser(description="Intermediate classification on arXiv data.")
    parser.add_argument('--dataset', type=str, default='dataset/Long-document-dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--log', type=str, default='log/intermediate_methods.log',
                        help='Path to the log file')
    parser.add_argument('--minimum_count_samples', type=str, nargs='+', default=["org", "minlength", "100", "140"],
                        metavar='N',
                        help='List of minimum counts of samples per class for balancing. Use "minlength" to automatically use the minimum class size. Use "org" to automatically use the orginal class size.')
    parser.add_argument('--models', type=str, nargs='+', default=['reg', 'svm', 'mlp', 'xgb'],
                        help='List of models to use: reg, svm, mlp, xgb')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")
    # Set up logging
    if not os.path.exists(os.path.dirname(args.log)):
        os.makedirs(os.path.dirname(args.log))

    # Update dataset path and log file if provided
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