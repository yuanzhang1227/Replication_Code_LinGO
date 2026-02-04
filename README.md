# Replication Code for "LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Uncivil Discourse"

This repository contains all Python scripts used for the experiments in the paper (for review purposes):  
**LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Uncivil Discourse**

- **Baseline_experiments.py**: runs zero-shot and chain-of-thought (CoT) prompting  
- **TextGrad_original.py**: runs direct optimization with TextGrad  
- **TextGrad_LinGO.py**: runs linguistic-graph optimization with TextGrad  
- **AdalFlow_original.py**: runs direct optimization with AdalFlow  
- **AdalFlow_LinGO.py**: runs linguistic-graph optimization with AdalFlow  
- **Dspy_original.py**: runs direct optimization with Dspy  
- **Dspy_LinGO.py**: runs linguistic-graph optimization with Dspy  
- **RAG_original.py**: runs direct optimization with RAG  
- **RAG_LinGO.py**: runs linguistic-graph optimization with RAG  
- **Fine_tuning_parameters.sh**: parameters used for LLM LoRA fine-tuning  

## Data Availability
We do not publicly share the datasets used in the experiments because they contain potentially identifiable personal information. The datasets can be provided privately to reviewers upon request during the review process.
