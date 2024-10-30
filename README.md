# EvoNB
## Code for the article `"EvoNB: A Protein Language Model-Based Workflow for Nanobody Mutation Prediction and Optimisation"`.

We fine-tuned [esm2_t33_650M_UR50D](https://github.com/facebookresearch/esm?tab=readme-ov-file) with approximately 7.66 million nanobody sequences. The full dataset and the fine-tuned models are available [here](https://huggingface.co/datasets/Dannyang/Nanobody_Sequence_Dataset).    
          
The following example outlines the steps for data preprocessing, model fine-tuning, and testing, along with guidance on using the model to predict mutations within nanobody sequences.
