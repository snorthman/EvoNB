import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO
import warnings
warnings.filterwarnings("ignore")

def predict_mutations(input_data, output_csv, model_checkpoints, device=None, n=5, input_type="sequence"):
    """
    input_data (str or list): Path to the input CSV file containing sequences, a list of sequences, or a FASTA file.
    output_csv (str): Path to the output CSV file to save the predicted mutations.
    model_checkpoints (list): List of model checkpoints to use for prediction.
    device (str, optional): Device to use for computation ('cuda' or 'cpu').
    n (int, optional): Threshold for mutation prediction. Defaults to 5.
    input_type (str, optional): Type of input data ('csv', 'sequence', or 'fasta').
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [AutoModelForMaskedLM.from_pretrained(checkpoint).to(device) for checkpoint in model_checkpoints]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints[0])

    if input_type == "csv":
        sequence_name = pd.read_csv(input_data).iloc[:, 0]
        sequence = pd.read_csv(input_data).iloc[:, -1]
        df = pd.DataFrame({'name': sequence_name, 'sequence': sequence})
    elif input_type == "sequence":
        sequence_name, sequence = zip(*input_data)
        df = pd.DataFrame({'name': sequence_name, 'sequence': sequence})
    elif input_type == "fasta":
        sequences = list(SeqIO.parse(input_data, "fasta"))
        sequence_name = [record.id for record in sequences]
        sequence = [str(record.seq) for record in sequences]
        df = pd.DataFrame({'name': sequence_name, 'sequence': sequence})
    else:
        raise ValueError("Invalid input_type. Must be 'csv', 'sequence', or 'fasta'.")

    dataset = Dataset.from_pandas(df)

    results = []
    mask_token = tokenizer.mask_token

    for seq in tqdm(dataset):
        seq_name = seq['name']

        original_sequence = list(seq['sequence'])
        mutation = []
        for i in range(len(original_sequence)):
            masked_sequence = original_sequence[:i] + [mask_token] + original_sequence[i+1:]
            inputs = tokenizer(''.join(masked_sequence), return_tensors='pt').to(device)

            sum_probs = None

            for model in models:
                outputs = model(**inputs)
                predictions = outputs.logits
                mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)
                mask_token_logits = predictions[0, mask_token_index, :]
                mask_token_probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)

                if sum_probs is None:
                    sum_probs = mask_token_probs
                else:
                    sum_probs += mask_token_probs

            avg_probs = sum_probs / len(models)

            origin_res_id = tokenizer.convert_tokens_to_ids(original_sequence[i])
            origin_res_id_prob = avg_probs[0, origin_res_id].item()

            topk_token_ids = torch.topk(avg_probs, k=1, dim=-1).indices[0].tolist()
            topk_tokens = tokenizer.convert_ids_to_tokens(topk_token_ids)
            topk_probs = avg_probs[0, topk_token_ids].tolist()
            token_probs = list(zip(topk_tokens, topk_probs))

            for token, prob in token_probs:
                if prob > (origin_res_id_prob * n):
                    mutation.append(f'{original_sequence[i]}{i+1}{token}_{round(prob, 4)}')

        results.append({'name': seq_name, 'mutation_probability': mutation, 'number': len(mutation)})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Predict mutations for sequences")
    parser.add_argument("-input_type", type=str, choices=["csv", "sequence", "fasta"], default="sequence", help="Type of input data ('csv', 'sequence', or 'fasta')")
    parser.add_argument("-input_data", type=str, help="Path to the input data (CSV or FASTA file). Not required if input_type is 'sequence'.")
    parser.add_argument("-name", type=str, nargs='*', help="Sequence name for direct input. Required if input_type is 'sequence'.")
    parser.add_argument("-sequence", type=str, nargs='*',help="Sequence for direct input. Required if input_type is 'sequence'.")
    parser.add_argument("-output_csv", type=str, required=True, help="Path to the output CSV file to save the predicted mutations")
    parser.add_argument("-model_checkpoints", type=str, required=True, help="List of model checkpoints to use for prediction. Multiple models can be connected with '+'.")
    parser.add_argument("-n", type=int, default=5, help="Threshold for mutation prediction. Defaults to 5")
    parser.add_argument("-device", type=str, default="cpu", help="Device to use for computation ('cuda' or 'cpu'). Defaults to 'cpu'")

    args = parser.parse_args()

    if args.input_type == "sequence":
        if not args.sequence or not args.name:
            raise ValueError("sequence and name are required when input_type is 'sequence'")
        input_data = list(zip(args.name, args.sequence))
    else:
        if not args.input_data:
            raise ValueError("input_data is required when input_type is 'csv' or 'fasta'")
        input_data = args.input_data

    model_checkpoints = args.model_checkpoints.split('+')
    print(f'Model {model_checkpoints} are being used.')
    predict_mutations(input_data, args.output_csv, model_checkpoints, args.device, args.n, args.input_type)

if __name__ == "__main__":
    main()
