# BT4012 Kaggle Competition: Detecting Illicit Bitcoin Transactions

This repository contains my notebook for the BT4012 kaggle competition. Each transaction is a node in a Bitcoin payment graph with 165 anonymised features. The task is to train on the labelled transactions from time steps 1-35 and rank the transactions from steps 36-49 by how likely they are to be illicit. The competition metric is ROC-AUC.

## Final models

Both final models use collective classification (CC): a first LightGBM model scores the labelled training and test transactions, and a second LightGBM model uses each transaction's features together with summaries of its neighbours' scores. Both use SMOTE on the second stage and 10 PCA components learned from the labelled and unlabelled training transactions.

| Submission file                        | Model                                         | Public leaderboard ROC-AUC |
| -------------------------------------- | --------------------------------------------- | -------------------------- |
| `final_cc_spread_smote_with_pca.csv`   | CC + spread of neighbour scores + SMOTE + PCA | 0.96801                    |
| `final_cc_pagerank_smote_with_pca.csv` | CC + score PageRank + SMOTE + PCA             | 0.96475                    |

## Reproducing the submissions quickly

`final_models.ipynb` trains only my 2 final models (same code and settings as the main notebook) and saves both submission files in `submissions/`. Its first cell installs the required packages from `requirements.txt`. It takes around 5-10 minutes and needs `train.csv`, `test.csv` and `txs_edgelist.csv` in the `data/` folder.

## Repository structure

```
e1155604.ipynb        full notebook (EDA, feature engineering, all experiments, final predictions)
final_models.ipynb    trains only the 2 final models and saves the submission files
requirements.txt      Python package dependencies
submissions/          the 2 final prediction files
    final_cc_spread_smote_with_pca.csv
    final_cc_pagerank_smote_with_pca.csv
README.md
.gitignore
```

## Notebook outline (`e1155604.ipynb`)

1. Exploratory data analysis
2. Data preprocessing
3. Baseline model and validation design (expanding-window forward validation)
4. Feature engineering (directional neighbour means, network measures from Lecture 8, PCA)
5. Feature selection
6. Model exploration (model families and class imbalance)
7. Graph models and ensembles (GNN, collective classification)
8. Hyperparameter tuning (Optuna)
9. Model selection
10. Prediction

## How to run

1. The competition data is not included in this repository. Download `train.csv`, `test.csv` and `txs_edgelist.csv` from the competition page and put them in a `data/` folder next to the notebooks:

   ```
   data/train.csv
   data/test.csv
   data/txs_edgelist.csv
   ```

2. Install the packages (the first cell of each notebook does this):

   ```
   pip install -r requirements.txt
   ```

3. Run `final_models.ipynb` to get only the submission files, or `e1155604.ipynb` from top to bottom for the full process. Both save the predictions in the `submissions/` folder.

Note: a full run of `e1155604.ipynb` should take 4-5 hours
