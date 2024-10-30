import torch
import pandas as pd
import numpy as np
import os
import warnings
import argparse

from torch.utils.data import DataLoader
from anarci import anarci
from tqdm import tqdm
from datasets import Dataset
from transformers import default_data_collator
from transformers import AutoTokenizer, AutoModelForMaskedLM

warnings.filterwarnings("ignore")

def predict_sequence_probabilities(input_csv, output_csv, model_checkpoints, device=None, batch_size=6):
    """
    Predict sequence probabilities and save the results to a CSV file.

    Args:
        input_csv (str): Path to the input CSV file containing sequences.
        output_csv (str): Path to the output CSV file to save the predicted probabilities.
        model_checkpoints (list): List of model checkpoints to use for prediction.
        device (str, optional): Device to use for computation ('cuda' or 'cpu').
        batch_size (int, optional): Batch size for processing sequences. Defaults to 6.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints[0])
    models = [AutoModelForMaskedLM.from_pretrained(checkpoint).to(device) for checkpoint in model_checkpoints]

    sequence_name = pd.read_csv(input_csv).iloc[:, 0]
    sequence = pd.read_csv(input_csv).iloc[:, -1]
    df = pd.DataFrame({'name': sequence_name, 'sequence': sequence})
    dataset = Dataset.from_pandas(df)

    cdr1 = (27, 37)
    cdr2 = (57, 64)
    cdr3 = (107, 117)

    mask_token = tokenizer.mask_token
    results = pd.DataFrame(columns=['Sequence Name', 'FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4', 'Average'])

    for seq in tqdm(dataset):
        input_ids = []
        attention_mask = []
        labels = []
        seq_name = seq['name']
        # numbering
        numbering, _, _ = anarci([("sequence", seq['sequence'])], scheme="imgt", output=False, allow='H')
        if numbering == [None]:
            print(f"Skipping {seq['name']} due to None numbering!")
            continue
        numbered_ids =  [residue[0][0] for residue in numbering[0][0][0] if residue[1] != '-']
        # print(numbered_ids)
        try:
            if numbered_ids.index(cdr1[0])-1 >= 0:
                fr1_pos = (0, numbered_ids.index(cdr1[0])-1)
            else:
                raise Exception("Condition not met")
        except Exception:
            fr1_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing FR1, set to NaN...')
        try:
            cdr1_pos = (numbered_ids.index(cdr1[0]), numbered_ids.index(cdr1[1]))
        except Exception:
            cdr1_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing CDR1, set to NaN...')
        try:
            fr2_pos = (numbered_ids.index(cdr1[1])+1, numbered_ids.index(cdr2[0])-1)
        except Exception:
            fr2_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing FR2, set to NaN...')
        try:
            cdr2_pos = (numbered_ids.index(cdr2[0]), numbered_ids.index(cdr2[1]))
        except Exception:
            cdr2_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing CDR2, set to NaN...')
        try:
            fr3_pos = (numbered_ids.index(cdr2[1])+1, numbered_ids.index(cdr3[0])-1)
        except Exception:
            fr3_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing FR3, set to NaN...')
        try:
            cdr3_pos = (numbered_ids.index(cdr3[0]), numbered_ids.index(cdr3[1]))
        except Exception:
            cdr3_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing CDR3, set to NaN...')
        try:
            if len(numbered_ids)-1 >= numbered_ids.index(cdr3[1])+1:
                fr4_pos = (numbered_ids.index(cdr3[1])+1, len(numbered_ids)-1)
            else:
                raise Exception("Condition not met")
        except Exception:
            fr4_pos = (np.nan, np.nan)
            #print(f'{seq_name} missing FR4, set to NaN...')

        original_sequence = list(seq['sequence'])
        for i in range(len(original_sequence)):
            masked_sequence = original_sequence[:i] + [mask_token] + original_sequence[i+1:]
            masked_tokened_sequence = tokenizer(''.join(masked_sequence), padding='max_length', max_length=170)
            origin_tokened_sequence = tokenizer(''.join(seq['sequence']), padding='max_length', max_length=170)
            origin_tokened_sequence['input_ids'] = [-100 if idx != i+1 else token_id for idx, token_id in enumerate(origin_tokened_sequence['input_ids'])]

            input_ids.append(masked_tokened_sequence['input_ids'])
            attention_mask.append(masked_tokened_sequence['attention_mask'])
            labels.append(origin_tokened_sequence['input_ids'])

        masked_df = pd.DataFrame({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': labels})
        masked_dataset = Dataset.from_pandas(masked_df)

        dataloader = DataLoader(
            masked_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )

        model_probs = []
        for model in models:
            model.eval()
            prob = []
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['labels'] = batch['labels'].long()
                outputs = model(**batch)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                mask = batch['labels'] != -100
                mask_indices = mask.nonzero(as_tuple=True)
                values = batch['labels'][mask_indices]

                indices = torch.arange(len(values))
                selected_probabilities = probabilities[indices, mask_indices[1], values]
                selected_probabilities_list = selected_probabilities.tolist()
                prob.extend(selected_probabilities_list)
            model_probs.append(prob)

        avg_prob = np.mean(model_probs, axis=0)

        fr1_prob = np.nan if np.isnan(fr1_pos).any() else round(sum(avg_prob[fr1_pos[0]:fr1_pos[1]+1]) / (fr1_pos[1] - fr1_pos[0] + 1), 4)
        cdr1_prob = np.nan if np.isnan(cdr1_pos).any() else round(sum(avg_prob[cdr1_pos[0]:cdr1_pos[1]+1]) / (cdr1_pos[1] - cdr1_pos[0] + 1), 4)
        fr2_prob = np.nan if np.isnan(fr2_pos).any() else round(sum(avg_prob[fr2_pos[0]:fr2_pos[1]+1]) / (fr2_pos[1] - fr2_pos[0] + 1), 4)
        cdr2_prob = np.nan if np.isnan(cdr2_pos).any() else round(sum(avg_prob[cdr2_pos[0]:cdr2_pos[1]+1]) / (cdr2_pos[1] - cdr2_pos[0] + 1), 4)
        fr3_prob = np.nan if np.isnan(fr3_pos).any() else round(sum(avg_prob[fr3_pos[0]:fr3_pos[1]+1]) / (fr3_pos[1] - fr3_pos[0] + 1), 4)
        cdr3_prob = np.nan if np.isnan(cdr3_pos).any() else round(sum(avg_prob[cdr3_pos[0]:cdr3_pos[1]+1]) / (cdr3_pos[1] - cdr3_pos[0] + 1), 4)
        fr4_prob = np.nan if np.isnan(fr4_pos).any() else round(sum(avg_prob[fr4_pos[0]:fr4_pos[1]+1]) / (fr4_pos[1] - fr4_pos[0] + 1), 4)
        average_prob = round(sum(avg_prob) / len(avg_prob), 4)

        current_result = pd.DataFrame({
            'Sequence Name': [seq_name],
            'FR1': [fr1_prob],
            'CDR1': [cdr1_prob],
            'FR2': [fr2_prob],
            'CDR2': [cdr2_prob],
            'FR3': [fr3_prob],
            'CDR3': [cdr3_prob],
            'FR4': [fr4_prob],
            'Average': [average_prob]
        })
        results = pd.concat([results, current_result])
    #    print(f'{seq_name} >>> FR1: {fr1_prob}, CDR1: {cdr1_prob}, FR2: {fr2_prob}, CDR2: {cdr2_prob}, FR3: {fr3_prob}, CDR3: {cdr3_prob}, FR4: {fr4_prob}, Average: {average_prob}')

    results.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Predict sequence probabilities and save the results to a CSV file.")
    parser.add_argument("-input_csv", type=str, required=True, help="Path to the input CSV file containing sequences.")
    parser.add_argument("-output_csv", type=str, required=True, help="Path to the output CSV file to save the predicted probabilities.")
    parser.add_argument("-models", type=str, required=True, help="List of model checkpoints to use for prediction. Multiple models can be connected with '+'.")
    parser.add_argument("-device", type=str, default="cpu", help="Device to use for computation ('cuda' or 'cpu'). Defaults to cpu.")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size for processing sequences. Defaults to 8.")

    args = parser.parse_args()
    model_checkpoints = args.models.split('+')
    predict_sequence_probabilities(args.input_csv, args.output_csv, model_checkpoints, args.device, args.batch_size)

if __name__ == "__main__":
    main()
