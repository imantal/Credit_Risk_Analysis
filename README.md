# Credit_Risk_Analysis
## Overview of the Analysis
The purpose of this project was to perform credit risk analysis using supervised machine learning models. A dataset from LendingClub, a peer-to-peer lending services company was used to train and evaluate the models. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Different sampling algorithms such as oversampling, undersampling, and the combination sampling were used to deal with the class imbalance in this study. Additionally, the ensemble learning technique was used to improve the performance of the model by reducing the bias.  Jupyther notebook and Python machine learning libraries were used in this project.
## Data overview and preparation
The original data had about 100 columns and 116k rows and included loan information such as the loan amount, loan term, interest rate and many more. The data cleaning included items such as dropping the null columns and rows, converting the interest rate to numerical values, and converting the target column values to low_risk and high_risk based on their values. Out of 68,817 applications, 347 of them were considered as high_risk (denoted as “0”) and the rest were low_risk loans (denoted as “1”). 
The loan_status column with content of either low_risk or high_risk was defined as the target and the rest of the columns was considered as the features. The data was split into training and testing subsets using the “train_test_split” function from the “sklearn library”.
## Oversampling algorithms 
The idea of the oversampling algorithm is simple and intuitive: If one class has too few instances in the training set, more instances from that class are chosen for the training as shown in Figure 1.  For this study, two oversampling algorithms were used to determine which algorithm results in the best performance. The data was oversampled using the naive random oversampling algorithm and the synthetic minority oversampling technique (SMOTE). In the naive random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. In SMOTE, new instances are created by interpolating the values of the neighbors.

<p align="center">

![image](https://user-images.githubusercontent.com/103223944/183333174-85cdb7dc-5dfa-4919-8b1e-6b9d405526a0.png)

Figure 1: Oversampling algorithms
  
 </p>
