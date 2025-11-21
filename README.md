# Credit Card Fraud Detection
## About the project
Due to the vast volume of daily transactions, financial institutions face significant challenges in detecting and preventing fraudulent card transactions. Traditional fraud detection systems often fail to adapt to the speed of the evolving fraud tactics, resulting in undetected fraud that not only inconvenience customers and merchants but can also lead to losses in financial institutions.
This project aims to develop a data-driven fraud detection system that is capable of accurately identifying fraudulent activities using card transactions. By leveraging data analysis and a range of algorithms, this projects aims to accurately identify fraudulent activities, strengthening financial risk management systems.

## Repository Structure
```
Credit-Card-Fraud-Detection/
├── Notebooks/
│   ├── EDA
│       └── EDA.ipynb                                   # EDA of engineered dataset
│   ├── Feature Engineering
│       ├── feature_eng.ipynb                           # preprocessing steps and feature engineering
│       └── feature_eng_rf.ipynb                        # preprocessing and feature engineering for random forest
│   └── Models
│       ├── Baseline model
│       └── Challenger models                          
│           ├── graphsage_lgbm_ensemble.ipynb           # graphsage and lgbm ensemble model
│           ├── hetero_graphsage_build_graph.py            
│           ├── hetero_graphsage_model_training.ipynb   # hetero graphsage model
│           └── Random Forest Classifier.ipynb          # random forest model
├── Data/
│   ├── raw/                                            # contains original datasets
│       ├── README.md                                   # Instructions for raw datasets
│       ├── sd254_cards.csv
│       └── sd254_users.csv
│   └── processed/  
│       ├── baseline_splits/                            # contains splits for baseline model
│       └── challenger_splits/                          # contains splits for challenger models
│           ├── rf_splits/                              # contains splits used in random forest model
│               ├── README.md                           # Instructions for random forest processed datasets
│               ├── test_y.csv
│               └── train_y.csv
├── GNN Data/
│ ├── best_search_model.pt                              # best model
│ ├── graph_15features_scaled_balanced.pt               # processed heterogeneous graph dataset
│ └── search_summary.json                               # best model hyperparameters
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
| Variable Name               | Data Type | Description                      | Example            |
|-----------------------------|-----------|----------------------------------|--------------------|
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
| `merchant_state_diff`       | Boolean   | Flag indicating whether the state of the merchant matches the state of the users    | True,False |
| `amount_is_refund`          | Integer   | Flag indicating if transaction values were negative | 0, 1   |
| `amount_log`                | Float     | Log of transaction amount  | 2.309561  |
| `von_mises_likelihood_card` | Float     | Measures how typical the transaction timing is for a given card  | 1.285982 |
| `Is Fraud?`                 | Integer   | Indicates if the transaction was fraudulent or not  | 0, 1 |

## Instructions on how to run the models
### Logistic Regression

### Random Forest

### Heterogeneous GraphSAGE


### LightGBM ensemble


## Evaluation of Models
