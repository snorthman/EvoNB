# EvoNB
### Code accompanying the article `"EvoNB: A Protein Language Model-Based Workflow for Nanobody Mutation Prediction and Optimisation"`.

We fine-tuned [esm2_t33_650M_UR50D](https://github.com/facebookresearch/esm?tab=readme-ov-file) with approximately 7.66 million nanobody sequences. The full [dataset](https://huggingface.co/datasets/Dannyang/Nanobody_Sequence_Dataset) and the fine-tuned [models](https://huggingface.co/Dannyang/EvoNB_1) are available at [huggingface](https://huggingface.co/).    

#### Environment Setup
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
wget -O /tmp/mmseqs-linux-avx2.tar.gz https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz /tmp/mmseqs-linux-avx2.tar.gz -C /tmp
sudo mv /tmp/mmseqs-linux-avx2 /usr/local/mmseqs
sudo ln -s /usr/local/mmseqs/bin/mmseqs /usr/local/bin/mmseqs
```
#
The following example outlines the steps for data preprocessing, model fine-tuning, and accuracy evaluation, along with guidance on using the model to predict mutations within nanobody sequences.

#### Data Preprocessing
```bash
python cluster.py -script_path ./dataset/mmseqs2.sh -input_faa ./data/example.fasta -mmseq_db_home ./data -db_name prot90 -min_cov 0.8 -min_id 0.9 -threads_num 8 -output_csv ./data/clu_rep.csv -validation_set_number 20 -test_set_number 20 -seed 42
```

#### Model Fine-Tuning
```bash
python train.py -train_file ./data/clu_rep_train.csv -valid_file ./data/clu_rep_validation.csv -test_file ./data/clu_rep_test.csv -model_checkpoint esm2_t33_650M_UR50D/ -datasets_dir ./data/example-dataset -output_model_dir model_tfd -seed 42 -device cuda
```

#### Accuracy Evaluation
```bash
# for single model
python predictive_accuracy.py -input_csv ../data/clu_rep_test.csv -output_csv ./out_pred.csv -models EvoNB_1 -device cuda
# for multiple models
python predictive_accuracy.py -input_csv ../data/clu_rep_test.csv -output_csv ./out_pred.csv -models EvoNB_1+EvoNB_2+EvoNB_3+EvoNB_4+EvoNB_5 -device cuda
```
Note: Using 5 models at the same time requires at least 24G of GPU memory!

#### Mutations Prediction
```bash
# for sequence
python get_mutation.py -input_type sequence -name seq1 -sequence "QVQLVESGGGLVQSGGSLRLSCAASGSIFRTTGMNWYRQTPEKQREWVALITSHGTTSYAASVEGRFTISRDSAGTTVYLQMNSLKPEDAGVYYCTTRGYWGQGTQVTVSS" -output_csv out_mut.csv -model_checkpoints EvoNB_1+EvoNB_2+EvoNB_3+EvoNB_4+EvoNB_5 -n 5 -device cuda
# for csv file
python get_mutation.py -input_type csv -input_data ./example_mut/example.csv -output_csv out_mut.csv -model_checkpoints EvoNB_1+EvoNB_2+EvoNB_3+EvoNB_4+EvoNB_5 -n 5 -device cuda
# for fasta file
python get_mutation.py -input_type fasta -input_data ./example_mut/example.fasta -output_csv out_mut.csv -model_checkpoints EvoNB_1+EvoNB_2+EvoNB_3+EvoNB_4+EvoNB_5 -n 5 -device cuda
```
