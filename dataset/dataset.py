import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from functools import partial
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore")


def prepare_datasets(
    train_file, 
    valid_file, 
    test_file, 
    model_checkpoint, 
    max_length=170, 
    num_proc=8, 
    out_dir='../data/example_datasets'
):
    """
    Prepare datasets for training and evaluation
    train_file: str, path to the training data
    valid_file: str, path to the validation data
    test_file: str, path to the test data
    model_checkpoint: str, model checkpoint to use
    max_length: int, max length of the input sequence
    num_proc: int, number of processes to use
    out_dir: str, path to save the prepared datasets
    """

    if os.path.exists(out_dir):
        print(f"{out_dir} already exists. Loading existing dataset.")
        dataset_dict = load_from_disk(out_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        return dataset_dict, model, tokenizer

    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # load the data
    train_sequence = pd.read_csv(train_file).iloc[:, -1]
    valid_sequence = pd.read_csv(valid_file).iloc[:, -1]
    test_sequence = pd.read_csv(test_file).iloc[:, -1]
    train_df = pd.DataFrame(train_sequence)
    valid_df = pd.DataFrame(valid_sequence)
    test_df = pd.DataFrame(test_sequence)
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    # tokenize the data
    def tokenize_function(sequence, max_length=max_length):
        sequences = list(sequence.values())[0]
        result = tokenizer(sequences, padding='max_length', max_length=max_length)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenize_function = partial(tokenize_function, max_length=max_length)
    train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["sequence"], num_proc=num_proc)
    valid_tokenized_datasets = valid_dataset.map(tokenize_function, batched=True, remove_columns=["sequence"], num_proc=num_proc)
    test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True, remove_columns=["sequence"], num_proc=num_proc)

    dataset_dict = DatasetDict({
        "train": train_tokenized_datasets,
        "validation": valid_tokenized_datasets,
        "test": test_tokenized_datasets
    })
    dataset_dict.save_to_disk(out_dir)

    return dataset_dict, model, tokenizer

def main():
    train_file = "../data/clu_rep_train.csv"
    valid_file = "../data/clu_rep_validation.csv"
    test_file = "../data/clu_rep_test.csv"
    model_checkpoint = "../esm2_t33_650M_UR50D"
    out_dir = '../data/example_datasets'
    dataset_dict, model, tokenizer = prepare_datasets(
        train_file=train_file, 
        valid_file=valid_file, 
        test_file=test_file, 
        model_checkpoint=model_checkpoint, 
        max_length=170, 
        num_proc=8, 
        out_dir=out_dir
    )

if __name__ == "__main__":
    main()
