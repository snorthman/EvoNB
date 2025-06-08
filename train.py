import os
import torch
import argparse
from dataset.dataset import prepare_datasets
from model.model import set_seed, train_model
import warnings
warnings.filterwarnings("ignore")

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    required_args = parser.add_argument_group("Required arguments")
    required_args.add_argument("-train_file", type=str, required=True, help="Path to the training data")
    required_args.add_argument("-valid_file", type=str, required=True, help="Path to the validation data")
    required_args.add_argument("-test_file", type=str, required=True, help="Path to the test data")
    required_args.add_argument("-model_checkpoint", type=str, required=True, help="Model checkpoint to use")
    required_args.add_argument("-datasets_dir", type=str, required=True, help="Path to save the prepared datasets")
    required_args.add_argument("-output_model_dir", type=str, required=True, help="Output directory name for the trained model")

    optional_args = parser.add_argument_group("Optional arguments")
    optional_args.add_argument("-max_length", type=int, default=170, help="Max length of the input sequence")
    optional_args.add_argument("-num_proc", type=int, default=8, help="Number of processes to use")
    optional_args.add_argument("-batch_size", type=int, default=16, help="Batch size for training")
    optional_args.add_argument("-mask_probability", type=float, default=0.15, help="Mask probability for MLM")
    optional_args.add_argument("-num_train_epochs", type=int, default=3, help="Number of training epochs")
    optional_args.add_argument("-learning_rate", type=float, default=5e-5, help="Learning rate")
    optional_args.add_argument("-weight_decay", type=float, default=0.001, help="Weight decay")
    optional_args.add_argument("-seed", type=int, default=42, help="Random seed")
    optional_args.add_argument("-device", type=str, default='cpu', help="Device to use for training")

    args = parser.parse_args()

    set_seed(args.seed)

    # get the dataset
    print("Preparing datasets...")
    dataset_dict, model, tokenizer = prepare_datasets(
        train_file=args.train_file, 
        valid_file=args.valid_file, 
        test_file=args.test_file, 
        model_checkpoint=args.model_checkpoint, 
        max_length=args.max_length, 
        num_proc=args.num_proc, 
        out_dir=args.datasets_dir
    )

    # train the model
    model.to(args.device)
    print(f'Will train on the {args.device}...')
    trainer = train_model(
        dataset_dict=dataset_dict, 
        model=model, 
        tokenizer=tokenizer, 
        output_dir_name=args.output_model_dir, 
        batch_size=args.batch_size, 
        mask_probability=args.mask_probability, 
        num_train_epochs=args.num_train_epochs, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        seed=args.seed
    )
    trainer.train()

if __name__ == "__main__":
    main()
