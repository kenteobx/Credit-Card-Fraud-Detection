# Credit Card Fraud Detection
## About the project
Due to the vast volume of daily transactions, financial institutions face significant challenges in detecting and preventing fraudulent card transactions. Traditional fraud detection systems often fail to adapt to the speed of the evolving fraud tactics, resulting in undetected fraud that not only inconvenience customers and merchants but can also lead to losses in financial institutions.
<br><br>
**This project aims to develop a data-driven fraud detection system that is capable of accurately identifying fraudulent activities using card transactions.** By leveraging data analysis and a range of algorithms, this project aims to accurately identify fraudulent activities, strengthening financial risk management systems.
<br><br>
Methods:
* Baseline Model: Logistic Regression
* Challenger Models: Random Forest, Heterogeneous GraphSAGE, GraphSAGE-LightGBM Ensemble 

## Repository Structure
```
Credit-Card-Fraud-Detection/
├── Notebooks/
│   ├── EDA
│   │   └── EDA.ipynb                                   # EDA of engineered dataset
│   ├── Feature Engineering
│   │   ├── feature_eng.ipynb                           # preprocessing steps and feature engineering
│   │   └── feature_eng_rf.ipynb                        # preprocessing and feature engineering for random forest
│   └── Models
│       ├── Baseline model
│       │   └── logistic_regression.ipynb               # baseline logistic regression model
│       └── Challenger models                          
│           ├── graphsage_lgbm_ensemble.ipynb           # graphsage and lgbm ensemble model
│           ├── hetero_graphsage_build_graph.py         # prepares dataset for hetero graphsage and ensemble model
│           ├── hetero_graphsage_model_training.ipynb   # hetero graphsage model
│           └── Random Forest Classifier.ipynb          # random forest model
├── Data/
│   ├── raw/                                            
│   │   ├── README.md                                   # instructions for raw datasets
│   │   ├── sd254_cards.csv                             # original cards dataset
│   │   └── sd254_users.csv                             # original users dataset
│   └── processed/  
│       ├── baseline_splits/                            # contains splits for baseline model
│       │   ├── test_X.csv
│       │   ├── test_y.csv
│       │   ├── train_X.csv
│       │   └── train_y.csv
│       └── challenger_splits/                          # contains splits for challenger models
│           ├── rf_splits/                              # contains splits used in random forest model
│           │   ├── README.md                           # instructions for random forest processed datasets
│           │   ├── test_y.csv
│           │   └── train_y.csv
│           └── original/                               # contains original splits for challenger models
│               ├── test_X.csv
│               ├── test_y.csv
│               ├── train_X.csv
│               └── train_y.csv
├── GNN Data/
│   ├── best_search_model.pt                            # best model
│   ├── graph_15features_scaled_balanced.pt             # processed heterogeneous graph dataset
│   └── search_summary.json                             # best model hyperparameters
├── README.md                                           # project documentation
├── eda_requirements.txt                                # Python dependencies for EDA.ipynb
└── requirements.txt                                    # Python dependencies for GNN
```

## Dataset
Data Source: [Kaggle Credit Card Transactions](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data)  

### Description: 
The dataset consists of:  
* `credit_card_transactions-ibm_v2.csv`: contains information on the card transactions e.g. merchant, transaction time etc.     
* `sd254_cards.csv`: contains information on the users' cards such as the card type, card brand, availability of the card on the dark web  
* `sd254_users.csv`: contains demographic and financial(e.g. yearly income, total debt) information on the users 

### Data Dictionary of Processed Dataset
| Variable Name         | Data Type | Description                      | Example            |
|-----------------------|-----------|----------------------------------|--------------------|
| `User`                      | Integer   | User ID              | 716      |
| `Card`                      | Integer   | User's nth card      | 2        |
| `Year`                      | Integer   | Year of transaction  | 2011     |
| `Month`                     | Integer   | Month of transaction | 12       |
| `Day`                       | Integer   | Day of transaction   | 4        |
| `Time`                      | String    | Transaction hour and minute | "13:53"  |
| `Amount`                    | Float     | Transaction amount   | 15.85       |
| `Use Chip`                  | String    | Transaction method (Online, Swipe or Chip) | "Online Transaction"     |
| `Merchant Name`             | Integer   | Merchant's unique identifier | 6780853441840436625  |
| `Merchant City`             | String    | City of merchant's store  | "Fremont"   |
| `Merchant State`            | String    | 2-letter state abbreviation of merchant's store | "TX"        |
| `Zip`                       | Float     | Zip code of merchant's store | 94536.0  |
| `MCC`                       | Integer   | Merchant Category Code based on merchant's type of goods or services | 7210  |
| `Errors?`                   | String    | Type of transaction error | "Insufficient Balance"   |
| `Zip_str`                   | String    | First 3 digits of merchant zip code  | "945"  |
| `DateTime`                  | String    | String of transaction date and time | "2011-12-04T13:53:00.000000"   |
| `Date`                      | String    | String of transaction date  | "2011-12-04"	  |
| `Hour`                      | Integer   | Transaction hour  | 13  |
| `User_card`                 | String    | Unique identifier of user and specific card used for transaction  | "835_3"  |
| `State`                     | String    | 2-letter abbreviation of user's home state     | "CA"    |
| `FICO Score`                | Integer   | User's credit score   | 798  |
| `Yearly Income - Person`    | Float     | User's yearly income  | 68765.0  |
| `Total Debt`                | Float     | User's total debt     | 75608.0  |
| `Num Credit Cards`          | Integer   | Number of credit cards owned by user | 5  |
| `Card Brand`                | String    | Brand of card used for transaction  | "Mastercard"   |
| `Credit Limit`              | Float     | Card's credit limit   | 39751.0  |
| `Card on Dark Web`          | String    | Whether the card is on the dark web    | "No"     |
| `merchant_state_diff`       | Boolean   | Flag indicating whether the state of the merchant matches the state of the users    | True, False |
| `amount_is_refund`          | Integer   | Flag indicating if transaction values were negative | 0, 1   |
| `amount_log`                | Float     | Log of transaction amount  | 2.309561  |
| `von_mises_likelihood_card` | Float     | Measures how typical the transaction timing is for a given card  | 1.285982 |
| `Is Fraud?`                 | Integer   | Indicates if the transaction was fraudulent or not  | 0, 1 |

## Instructions on how to run the models
1. Clone repository
```
git clone https://github.com/kenteobx/Credit-Card-Fraud-Detection
cd Credit-Card-Fraud-Detection
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Download dataset

4. Open and execute `feature_eng.ipynb` to obtain train and test splits
```
Notebooks/Feature Engineering/feature_eng.ipynb
```
### Logistic Regression
#### Motivation
We used a logistic regression model as the baseline model as it is a classification method that has been widely explored in fraud detection. Furthermore, it does not make any assumptions about the distribution of classes in the feature space, making it more flexible than linear regression. Coupled with its simplicity, high interpretability and efficiency, logistic regression makes an appropriate choice for a baseline. 

### Random Forest
#### Motivation
A Random Forest Classifier was selected as a challenger model because it can address the challenges that come with transactional datasets such as class imbalance, mixed features types and nonlinear fraud patterns. Fraudulent transactions are rare and depend on complex nonlinear behavioural and contextual features such as transaction frequency, amount, location and merchant. For the credit card transactions dataset, it is severely imbalanced, with only 0.12% of the transactions being fraudulent. Moreover, an analysis of the correlation between the features, consisting of both engineered and existing, shows that the features are mostly weakly correlated with the target variable (“Is Fraud?”). This suggests that fraud cannot be easily separated using linear relationships and most likely depends on nonlinear interactions. Hence, a Random Forest Classifier is suitable to be used as a challenger model.

<u>Steps to run the Random Forest model</u>
1. Follow the above steps in `Instructions on how to run the models` section

2. Open and execute `feature_eng_rf.ipynb` to obtain the processed datasets for the random forest model
```
Notebooks/Feature Engineering/feature_eng_rf.ipynb
```
3. Open and execute `Random Forest Classifier.ipynb` to run the random forest model
```
Notebooks/Models/Challenger models/Random Forest Classifier.ipynb
```

### Heterogeneous GraphSAGE
Traditional fraud detection models often fail to capture sophisticated fraud rings where attackers mimic normal behaviour to blend in. However, these fraudulent activities frequently rely on a shared, underlying infrastructure. For instance, a group of fraudulent users might be linked to the same devices, IP addresses, or colluding merchants.
A GNN is well suited for this setting because it learns embeddings (feature representations) for each entity not only from its own attributes, but also from the attributes and labels of its neighbours. This allows the model to detect suspicious communities, such as a dense cluster of users and merchants sharing a small set of devices or locations, patterns that are invisible to a standard transaction-level model.


### GraphSAGE - LightGBM ensemble
#### Motivation
GNNs excel at learning expressive representations by aggregating neighbourhood information, but their final prediction layers are typically shallow (a linear layer or small MLP). In contrast, gradient-boosted trees such as LightGBM handle tabular decision boundaries extremely well.

To combine these strengths, a stacking ensemble was used: the Heterogeneous GraphSAGE model serves purely as a feature extractor, and LightGBM performs the final fraud classification.



## Evaluation of Models

|Model|ROC-AUC|PR-AUC|Comments|
|-----|-------|------|--------|
|Logistic Regression|0.5341|0.0014|Baseline is only slightly better than random guessing, with extremely low PR-AUC reflecting inability to identify fraud cases due to severe class imbalance.|
|Random Forest|0.9823|0.6590|Achieves a strong ranking performance and solid fraud-detection effectiveness despite severe class imbalance. |
|Heterogeneous GraphSAGE|0.8853|0.5840|Captures user–merchant behavioural structure effectively and outperforms all tabular baselines. Provides strong ranking performance under severe class imbalance, with a clear PR-AUC lift despite low fraud prevalence.|
|GraphSAGE-LightGBM Ensemble|0.9299|0.7374|Leverages the GNN’s structural embeddings and substantially boosts predictive accuracy. LightGBM models the non-linear fraud patterns that the GNN head alone cannot, resulting in the highest overall ROC-AUC and PR-AUC.|
