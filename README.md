# TabPFN vs XGBoost on Tabular Datasets

## Overview
This project looks at how well TabPFN (Tabular Prior-Data Fitted Network) performs compared to XGBoost on three different datasets:  
1. Mushroom Classification  
2. Heart Disease Prediction  
3. Adult Income Classification  

The experiments draw inspiration from the paper:  
**Prior-Data Fitted Networks (TabPFN)**, Nature, 2024  
https://www.nature.com/articles/s41586-024-08328-6  

That paper showed that a foundation model for tabular data can reach competitive or better performance than well-tuned machine learning models without any hyperparameter tuning. This is especially true for small-data situations.

## Objective
The aims of this project are to:  
- Capture the essence of the original research by comparing TabPFN to a strong baseline (XGBoost) on datasets of varying sizes and complexities.  
- Assess performance without hyperparameter tuning for either model.  
- Identify where TabPFN performs well, where XGBoost is more effective, and where both models are comparable.

## Notebooks
- Mushroom_tabpfn_vs_xgb.ipynb  
- Heart_Disease_tabpfn_vs_xgb.ipynb  
- Adult_Income_tabpfn_vs_xgb.ipynb  

Each notebook:  
- Loads and preprocesses the dataset  
- Trains TabPFN (zero-shot)  
- Trains XGBoost (default parameters)  
- Reports accuracy, precision, recall, and F1-score  
- Visualizes feature importance for XGBoost  

## Experimental Setup
**TabPFN**  
- Used in zero-shot mode (no tuning)  
- CPU for smaller datasets; GPU (Colab Pro L4) for the Adult Income dataset  
- Downsampled to 2,000 rows for Adult Income to fit TabPFN’s small-data focus and architectural limits  

**XGBoost**  
- Trained on the full dataset without tuning  
- Default parameters with eval_metric='logloss'  

**Evaluation Metrics**  
- Accuracy  
- Precision, Recall, F1-score (macro and weighted averages)  

### TabPFN Limitations
- TabPFN is best for small datasets (about ≤3,000 training samples, ≤100 features).  
- Large datasets are limited by the model’s transformer attention mechanism (O(N²) complexity).  
- For the Adult Income dataset, the training set was downsampled to 2,000 rows to adhere to these limits. XGBoost trained on the full dataset.  

### GPU Usage
- The GPU (Colab Pro L4) was used for the Adult Income experiment.  
- TabPFN benefits from GPU acceleration mainly during batch prediction. The GPU does not eliminate the small-data limitation but can speed up training and inference.  

### Feature Importance
- Feature importance visualizations are provided for XGBoost.  
- TabPFN currently does not have built-in feature importance outputs like tree-based models, so a direct interpretability comparison is not available.  

### Statistical Significance
- Results are based on a single train-test split.  
- A thorough evaluation would include k-fold cross-validation or multiple random train-test splits to measure variance in metrics and statistical significance.

## Results Summary

| Dataset        | Size (rows) | TabPFN Accuracy | XGBoost Accuracy | Notes |
|----------------|-------------|----------------|------------------|-------|
| Mushroom       | ~8,100      | 1.000           | 1.000            | Dataset is perfectly separable; both models perform identically |  
| Heart Disease  | 303         | 0.783           | 0.700            | TabPFN outperforms XGBoost in small-data setting; better recall on positive (disease) class |  
| Adult Income   | 32,000+ full / 2,000 sample | 0.848 | 0.860 | TabPFN trained only on 2,000-row sample; XGBoost on full dataset. Highlights small-data ability vs. large-data scalability |  

## Key Insights
1. Mushroom dataset — Both models achieve perfect classification because of strong signals in features like odor and spore-print-color.  
2. Heart Disease dataset — TabPFN has an edge in small-data settings, outperforming XGBoost in most metrics without tuning.  
3. Adult Income dataset — XGBoost takes advantage of the full dataset size, while TabPFN stays competitive when trained on a much smaller subset.  

## Interpretation
These experiments support the findings in the TabPFN paper:  
- TabPFN works best in small, structured datasets without tuning.  
- XGBoost remains a solid choice in large-data, high-variance situations.  
- On clean datasets with strong signals, both models can reach perfect or nearly perfect accuracy.  

## How to Reproduce
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/tabpfn_vs_xgb.git  
   cd tabpfn_vs_xgb  
   ```  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt  
   ```  
3. Run each notebook in Jupyter or Google Colab.  

## References
- Original Paper: https://www.nature.com/articles/s41586-024-08328-6  
- TabPFN GitHub: https://github.com/automl/TabPFN  
- XGBoost Documentation: https://xgboost.readthedocs.io/
