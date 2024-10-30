import torch
import os
import random
import numpy as np
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(
    dataset_dict, 
    model, 
    tokenizer, 
    output_dir_name = 'example_model', 
    batch_size=16, 
    mask_probability=0.15, 
    num_train_epochs=3, 
    learning_rate=5e-5, 
    weight_decay=0.001, 
    seed=42
):
    '''
    dataset_dict: DatasetDict, dataset dictionary
    model: PreTrainedModel, model to train
    tokenizer: PreTrainedTokenizer, tokenizer to use
    output_dir_name: str, output directory name
    batch_size: int, batch size
    mask_probability: float, probability of masking tokens in the input
    num_train_epochs: int, number of training epochs
    learning_rate: float, learning rate
    weight_decay: float, weight decay
    seed: int, random seed
    '''

    # load the datasets
    train_tokenized_datasets = dataset_dict["train"]
    valid_tokenized_datasets = dataset_dict["validation"]

    # set the training arguments
    logging_steps = len(train_tokenized_datasets) // batch_size
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mask_probability)

    training_args = TrainingArguments(
        output_dir=output_dir_name,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=valid_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer

def main():
    seed=42
    set_seed(seed)
    dataset_dict = load_from_disk('../data/example_datasets')
    model_checkpoint = "../esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    output_dir_name='../model_ft'
    trainer = train_model(
        dataset_dict=dataset_dict, 
        model=model, 
        tokenizer=tokenizer, 
        output_dir_name=output_dir_name, 
        batch_size=16, 
        mask_probability=0.15, 
        num_train_epochs=3, 
        learning_rate=5e-5, 
        weight_decay=0.001, 
        seed=seed
    )
    trainer.train()

if __name__ == "__main__":
    main()
