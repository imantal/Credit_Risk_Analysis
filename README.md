# Credit_Risk_Analysis
## Overview of the Analysis
The purpose of this project was to perform credit risk analysis using supervised machine learning models. A dataset from LendingClub, a peer-to-peer lending services company was used to train and evaluate the models. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Different sampling algorithms such as oversampling, undersampling, and the combination sampling were used to deal with the class imbalance in this study. Additionally, the ensemble learning technique was used to improve the performance of the model by reducing the bias.  Jupyther notebook and Python machine learning libraries were used in this project.
## Data overview and preparation
The original data had about 100 columns and 116k rows and included loan information such as the loan amount, loan term, interest rate and many more. The data cleaning included items such as dropping the null columns and rows, converting the interest rate to numerical values, and converting the target column values to low_risk and high_risk based on their values. Out of 68,817 applications, 347 of them were considered as high_risk (denoted as “0”) and the rest were low_risk loans (denoted as “1”). 
The loan_status column with content of either low_risk or high_risk was defined as the target and the rest of the columns was considered as the features. The data was split into training and testing subsets using the “train_test_split” function from the “sklearn library”.
## Oversampling algorithms 
The idea of the oversampling algorithm is simple and intuitive: If one class has too few instances in the training set, more instances from that class are chosen for the training as shown in Figure 1.  For this study, two oversampling algorithms were used to determine which algorithm results in the best performance. The data was oversampled using the naive random oversampling algorithm and the synthetic minority oversampling technique (SMOTE). In the naive random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. In SMOTE, new instances are created by interpolating the values of the neighbors.

<div align="center"> 

![image](https://user-images.githubusercontent.com/103223944/183333174-85cdb7dc-5dfa-4919-8b1e-6b9d405526a0.png)

Figure 1: Oversampling algorithms
  
<div align="left"> 

For each algorithm, the following steps were followed:
1.	View the count of the target classes using Counter from the “collections” library.
2.	Use the resampled data to train a logistic regression model.
3.	Calculate the balanced accuracy score from “sklearn.metrics”.
4.	Print the confusion matrix from “sklearn.metrics”.
5.	Generate a classification report using the “imbalanced_classification_report” from “imblearn”.
For each algorithm, the model performance was evaluated using the balanced accuracy score, confusion matrix and the imbalanced classification report as shown in Figure 2 and Figure 3. 
Both algorithms showed a relatively low accuracy score of about 64%. They both seemed to perform poorly in dealing with the minority class (high-risk loans) with low F1 scores of 0.02. The low F1 score was mainly due to the very low precision score of 0.01 indicating a large number of false-positive outputs. Both models showed similar recall score of 0.6 indicating that 40% of the high-risk loans were falsely predicted as low-risk loans. That essentially means 40% of the unqualified applicants will be given the loan. On the other hand, both models did a much better job predicting the low-risk loans with precision score of 1.00, recall score of 0.7 and F1 score of 0.81.

  
 <div align="center"> 
    
![image](https://user-images.githubusercontent.com/103223944/183334727-31bcc50d-7ccc-4f25-ba02-6b09023ab926.png)
  
Figure 2:  Model performance for naive random oversampling algorithm  

   
![image](https://user-images.githubusercontent.com/103223944/183334740-642d3f14-a3db-4230-b47c-ddb46492bed9.png)

 Figure 3:  Model performance for SMOTE oversampling algorithm 
  
 <div align="left">   
 
## Undersampling algorithms 
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. For this study, cluster centroid undersampling technique was used. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

<div align="center"> 
   
![image](https://user-images.githubusercontent.com/103223944/183334867-79e5db13-1b1f-4383-8709-01e311720255.png)

Figure 4:  Undersampling algorithm 
  
<div align="left">    

Similar to the previous section, the model performance was evaluated using the balanced accuracy score, confusion matrix and the imbalanced classification report as shown in Figure 5 
The cluster centroid undersampling algorithm showed a relatively low accuracy score of 53% which is lower than the ones observed with the oversampling algorithms discussed in previous section. The model performed poorly in dealing with the minority class (high-risk loans) with low F1 score of 0.01 which is mainly due to the low precision score of 0.01.  The recall score of 0.61 indicates that 39% of the high-risk loans were falsely predicted as low-risk loans. On the other hand, the model showed relatively a better performance dealing with the low-risk loans with precision score of 1.00 and F1 score of 0.62. Overall, the performance of the oversampling techniques discussed in the previous section was better than the undersampling technique. 

 <div align="center">  
  
![image](https://user-images.githubusercontent.com/103223944/183334967-f248ad99-f834-4350-b6e9-6dc5d8f0ee5d.png)
  
Figure 5:  Model performance for cluster centroid undersampling algorithm 
  
<div align="left">  
   
