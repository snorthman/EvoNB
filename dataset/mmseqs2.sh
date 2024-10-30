# Place of fasta and database
input_faa=./data/example.fasta
mmseq_db_home=./data

DB_name=prot90
DB_path=${mmseq_db_home}/${DB_name}

min_cov=0.8
min_id=0.9
threads_num=8

# 1. fasta to DB format
mkdir ${DB_path}
tmp_folder=${DB_path}/tmp
mkdir ${tmp_folder}
DB=${DB_path}/${DB_name}

# mmseqs createdb proteome.fasta DB
mmseqs createdb ${input_faa} ${DB}

# 2. Clustering
DB_clu_folder=${DB_path}/clu
mkdir ${DB_clu_folder}
DB_clu=${DB_clu_folder}/clu

# mmseqs cluster DB DB_clu tmp
mmseqs cluster ${DB} ${DB_clu} ${tmp_folder} --cov-mode 0 -c ${min_cov} --min-seq-id ${min_id} --threads $threads_num

# 3. Extract tsv
# mmseqs createtsv DB DB DB_clu clustered.tsv
mmseqs createtsv ${DB} ${DB} ${DB_clu} ${DB_path}/clustered.tsv

# 4. Extract fasta
# mmseqs createseqfiledb DB DB_clu DB_clu_seq
# mmseqs result2flat DB DB DB_clu_seq DB_clu_seq.fasta --use-fasta-header
DB_clu_seq_folder=${DB_path}/clu_seq
mkdir ${DB_clu_seq_folder}
DB_clu_seq=${DB_clu_seq_folder}/clu_seq

mmseqs createseqfiledb ${DB} ${DB_clu} ${DB_clu_seq}
mmseqs result2flat ${DB} ${DB} ${DB_clu_seq} ${DB_clu_seq_folder}/clu_seq.fasta --use-fasta-header

# 5. Extract representative sequence 
# mmseqs createsubdb DB_clu DB DB_clu_rep
# mmseqs convert2fasta DB_clu_rep DB_clu_rep.fasta
DB_clu_rep_folder=${DB_path}/clu_rep
mkdir ${DB_clu_rep_folder}
DB_clu_rep=${DB_clu_rep_folder}/clu_rep

mmseqs createsubdb ${DB_clu} ${DB} ${DB_clu_rep}
mmseqs convert2fasta ${DB_clu_rep} ${DB_clu_rep_folder}/clu_rep.fasta
cp ${DB_clu_rep_folder}/clu_rep.fasta $mmseq_db_home
