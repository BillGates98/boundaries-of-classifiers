# Boundaries-of-classifiers

To reproduce the evaluations you just have to execute the two commands below on the data whose alignment vectors have already been generated with the embeddings.

> Download the data and place them in the suitable directory and run the experiments
  - Data Checkpoints [([Download](https://drive.google.com/drive/folders/1VsLP8PD3puRpQTNSzoS1e4zvGysRGuh_?usp=sharing))]

** We kindly request you to install the python(>= 3.8) dependencies gradually if missing. **
- Dependancies
  
  `$ python -m pip install -r ./requirements.txt`

- Run data on classifiers

  `$ sh ./job_sota.sh`

- Run histogram to draw figures

  `$ sh ./histogram.sh`



