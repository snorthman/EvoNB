import os
import subprocess
import argparse
from dataset.utils import fasta_to_csv, split_dataset

def Set_mmseqs2_parameters(input_faa, mmseq_db_home, db_name, min_cov, min_id, threads_num, script_path):
    '''
    input_faa: input fasta file path
    mmseq_db_home: mmseq2 database home path
    DB_name: database name
    min_cov: minimum coverage
    min_id: minimum identity
    threads_num: number of threads
    script_path: mmseqs2 script path
    '''
    with open(script_path, 'r') as file:
        script_content = file.readlines()

    for i, line in enumerate(script_content):
        if line.startswith("input_faa="):
            script_content[i] = f"input_faa={input_faa}\n"
        elif line.startswith("mmseq_db_home="):
            script_content[i] = f"mmseq_db_home={mmseq_db_home}\n"
        elif line.startswith("DB_name="):
            script_content[i] = f"DB_name={db_name}\n"
        elif line.startswith("min_cov="):
            script_content[i] = f"min_cov={min_cov}\n"
        elif line.startswith("min_id="):
            script_content[i] = f"min_id={min_id}\n"
        elif line.startswith("threads_num="):
            script_content[i] = f"threads_num={threads_num}\n"

    with open(script_path, 'w') as file:
        file.writelines(script_content)

    os.chmod(script_path, 0o755)
    subprocess.run([script_path], shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Set parameters for mmseqs2 and process datasets")
    parser.add_argument("-script_path", type=str, required=True, help="Path to the mmseqs2 script")
    parser.add_argument("-input_faa", type=str, required=True, help="Path to the input fasta file")
    parser.add_argument("-mmseq_db_home", type=str, required=True, help="Path to the mmseqs2 database home")
    parser.add_argument("-db_name", type=str, required=True, help="Name of the database")
    parser.add_argument("-min_cov", type=float, required=True, help="Minimum coverage")
    parser.add_argument("-min_id", type=float, required=True, help="Minimum identity")
    parser.add_argument("-threads_num", type=int, required=True, help="Number of threads")
    parser.add_argument("-output_csv", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("-validation_set_number", type=int, required=True, help="Number of samples in the validation set")
    parser.add_argument("-test_set_number", type=int, required=True, help="Number of samples in the test set")
    parser.add_argument("-seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    Set_mmseqs2_parameters(args.input_faa, args.mmseq_db_home, args.db_name, args.min_cov, args.min_id, args.threads_num, args.script_path)
    fasta_to_csv(args.input_faa, args.output_csv)

    train_set, validation_set, test_set = split_dataset(args.output_csv, validation_set_number=args.validation_set_number, test_set_number=args.test_set_number, seed=args.seed)
    train_set.to_csv(args.output_csv.replace(".csv", "_train.csv"), index=False)
    validation_set.to_csv(args.output_csv.replace(".csv", "_validation.csv"), index=False)
    test_set.to_csv(args.output_csv.replace(".csv", "_test.csv"), index=False)

if __name__ == "__main__":
    main()
