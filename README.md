# gec


### GEC Module

#### Dataset Preparation

- Raw datasets can be downloaded from the [BEA-2019 Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/) and [HuggingFace — nusnlp/NUCLE](https://huggingface.co/datasets/nusnlp/NUCLE).
- To pre-process raw parallel files into GECToR format: `python utils/preprocess_data.py -s SRC -t TGT -o OUT`

#### Training and Evaluation

The `GEC/notebooks/` directory contains two notebooks:

- one for **Training** (`training_gec.ipynb`) — three-stage GECToR curriculum
- one for **Evaluation** (`dev_gec.ipynb`) — per-stage inference and ERRANT scoring, including ensemble

The notebooks are designed to make experimentation easier by allowing common settings such as dataset paths and the number of epochs to be adjusted, reducing the need to repeatedly modify the main files.

During evaluation, `dev_gec.ipynb` covers:

- **Single-model inference** — runs per stage (Stage 1, 2, 3) for a selected encoder
- **Ensemble inference** — 3-way majority voting across TinyBERT, ALBERT, and MobileBERT, with MobileBERT as tiebreaker
- **ERRANT scoring** — scores predictions against the BEA-2019 dev gold M2 file (Precision, Recall, F0.5)

Trained model weights are not included in this repository but can be accessed via [Kaggle Checkpoints](https://www.kaggle.com/datasets/roseanncaguilar/checkpoints). Result notebooks and prediction outputs per model are saved under `GEC/results/`.





Trained model weights are not included in this repository but can be accessed via [Dataset Preparation](#dataset-preparation) section. Result notebooks and prediction outputs per model are saved under `GEC/results/`.
Trained model weights are not included in this repository but can be accessed via [`roseannnnnaguilar/gector-shared-updated`](https://www.kaggle.com/datasets/roseannnnnaguilar/gector-shared-updated). Result notebooks and prediction outputs per model are saved under `GEC/results/`.

Trained model weights are not included in this repository but can be accessed via 
 Pre-processed files ready for training are available at [`roseannnnnaguilar/gector-shared-updated`](https://www.kaggle.com/datasets/roseannnnnaguilar/gector-shared-updated) (requires access).
The notebooks are designed to simplify experimentation by exposing commonly adjusted parameters such as dataset paths, number of epochs, etc., to eliminate the need to repeatedly modify the main files.
Each notebook provides a complete workflow and is designed to run on Kaggle (T4 × 2) or Google Colab.
