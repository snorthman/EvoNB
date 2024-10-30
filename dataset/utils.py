from Bio import SeqIO
import pandas as pd

def fasta_to_csv(fasta_filename, csv_filename):
    records = list(SeqIO.parse(fasta_filename, "fasta"))
    data = {"id": [record.id for record in records],
            "sequence": [str(record.seq) for record in records]}
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)

def split_dataset(csv_filename, validation_set_number=50000, test_set_number=50000, seed=42):
    df = pd.read_csv(csv_filename)
    validation_set = df.sample(n=validation_set_number, random_state=seed)
    test_set = df.drop(validation_set.index).sample(n=test_set_number, random_state=seed)
    train_set = df.drop(validation_set.index).drop(test_set.index)
    return train_set, validation_set, test_set

