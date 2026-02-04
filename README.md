# Replication Code for "LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Online Uncivil Discourse"

This repository contains all Python scripts used for the experiments in the paper (for review purposes):  
**LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Online Uncivil Discourse**

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
Due to privacy considerations and platform data-sharing restrictions, we do not release raw Twitter/X content or user identifiers. We share only derived annotations (incivility category labels, intention labels, and reasoning paths), which do not enable re-identification. 


