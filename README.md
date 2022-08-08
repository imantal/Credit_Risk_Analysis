# Credit_Risk_Analysis
## Overview of the Analysis

<div align="justify"> 

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

<div align="justify">   
  
For each algorithm, the model performance was evaluated using the balanced accuracy score, confusion matrix and the imbalanced classification report as shown in Figure 2 and Figure 3. 
Both algorithms showed a relatively low accuracy score of about 64%. They both seemed to perform poorly in dealing with the minority class (high-risk loans) with low F1 scores of 0.02. The low F1 score was mainly due to the very low precision score of 0.01 indicating a large number of false-positive outputs. Both models showed similar recall score of 0.6 indicating that 40% of the high-risk loans were falsely predicted as low-risk loans. That essentially means 40% of the unqualified applicants will be given the loan. On the other hand, both models did a much better job predicting the low-risk loans with precision score of 1.00, recall score of 0.7 and F1 score of 0.81.

  
 <div align="center"> 
    
![image](https://user-images.githubusercontent.com/103223944/183334727-31bcc50d-7ccc-4f25-ba02-6b09023ab926.png)
  
Figure 2:  Model performance for naive random oversampling algorithm  

   
![image](https://user-images.githubusercontent.com/103223944/183334740-642d3f14-a3db-4230-b47c-ddb46492bed9.png)

 Figure 3:  Model performance for SMOTE oversampling algorithm 
  
 <div align="left">   
 
## Undersampling algorithms 

<div align="justify"> 
   
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. For this study, cluster centroid undersampling technique was used. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

<div align="center"> 
   
![image](https://user-images.githubusercontent.com/103223944/183334867-79e5db13-1b1f-4383-8709-01e311720255.png)

Figure 4:  Undersampling algorithm 
  
<div align="left">    

<div align="justify">   
  
Similar to the previous section, the model performance was evaluated using the balanced accuracy score, confusion matrix and the imbalanced classification report as shown in Figure 5. 
The cluster centroid undersampling algorithm showed a relatively low accuracy score of 53% which is lower than the ones observed with the oversampling algorithms discussed in previous section. The model performed poorly in dealing with the minority class (high-risk loans) with low F1 score of 0.01 which is mainly due to the low precision score of 0.01. The recall score of 0.61 indicates that 39% of the high-risk loans were falsely predicted as low-risk loans. On the other hand, the model showed relatively a better performance dealing with the low-risk loans with precision score of 1.00 and F1 score of 0.62. Overall, the performance of the oversampling techniques discussed in the previous section was better than the undersampling technique. 

 <div align="center">  
  
![image](https://user-images.githubusercontent.com/103223944/183334967-f248ad99-f834-4350-b6e9-6dc5d8f0ee5d.png)
  
Figure 5:  Model performance for cluster centroid undersampling algorithm 
  
<div align="left">  
  
## Combination sampling algorithms 

<div align="justify">   
  
Combination sampling, SMOTEENN, combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms in two-step process:
1.	Oversample the minority class with SMOTE.
2.	Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

Similar to the previous sections, the model performance was evaluated using the balanced accuracy score, confusion matrix and the imbalanced classification report as shown in Figure 6.
The SMOTEENN algorithm showed a relatively low accuracy score 64%. The model performed poorly in dealing with the minority class (high-risk loans) with a low F1 score of 0.02 which is mainly due to low precision score of 0.01.  The recall score of 0.7 indicates that 30% of the high-risk loans were falsely predicted as low-risk loans. This means that essentially 30% of the unqualified applicants will be given the loan. The performance of this model is better compared to the oversampling and undersampling techniques discussed in the previous sections. On the other hand, the model showed relatively a better performance predicting the low-risk loans with precision score of 1.00 and F1 score of 0.73. Overall, the performance of the SMOTEENN technique was better than the undersampling and oversampling algorithms in dealing with the high-risk loans.  

 <div align="center">  
  
![image](https://user-images.githubusercontent.com/103223944/183335076-33e5369d-0608-48de-9432-b678089141ee.png)

Figure 6:  Model performance for SMOTEENN algorithm    
   
<div align="left"> 
  
## Ensemble learning 
  
<div align="justify">  
  
The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model. For this study, two ensemble learning algorithms i.e. the Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier from the “imblearn.ensemble” library were used.  Each classifier used total number of 100 tress in the model.

 <div align="center">  
   
![image](https://user-images.githubusercontent.com/103223944/183335255-816a3bef-23c4-40cf-bfd8-2cbbdbc1e490.png)
   
 Figure 7:  Ensemble learning algorithm 
   
 <div align="left"> 
   
<div align="justify"> 
   
Similar to the previous sections, the model performance was evaluated using the balanced accuracy score, confusion matrix and the imbalanced classification report as shown in Figure 8.

Both algorithms showed higher accuracy score compared to the ones discussed in previous sections. It is obvious that the ensemble learning technique was resulted in more accurate models with balanced accuracy scores of 79% and 93%.  Both models seemed to perform poorly in dealing with the minority class (high-risk loans) with low F1 scores of 0.07 and 0.14. The low F1 score was mainly due to the very low precision scores of 0.04 and 0.07 indicating a large number of false-positive outputs. However, the recall score was very high for the model with the Easy Ensemble AdaBoost classifier indicating a very low number of false-negative outputs. The recall score of 0.91 means that only 9% of the high-risk loans was falsely predicted as low-risk loans. Both models performed extremely well predicting the low-risk loans with high F1, recall and precision scores.  

 <div align="center">     
  
![image](https://user-images.githubusercontent.com/103223944/183335355-d5a893dd-55b9-4867-a6c3-46603abcf722.png)
  
Figure 8:  Model performance for Balanced Random Forest Classifier algorithm   
  
![image](https://user-images.githubusercontent.com/103223944/183335382-fb9d9f88-1188-43b9-ad11-c89fda8db8cc.png)

Figure 9:  Model performance for Easy Ensemble AdaBoost classifier algorithm 
   
<div align="left">    

## Summary and conclusion

<div align="justify">   
  
Different machine learning models were trained and then used to predict the credit risk based on a dataset from LendingClub, a peer-to-peer lending services company. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, oversampling, undersampling and combination sampling techniques were used to create balanced training datasets. Two additional machine learning models were defined based on the ensemble learning technique to reduce the bias.  The balanced accuracy scores, the precision and recall scores of all six machine learning models are summarized in Table 1 and Table 2.
 
The balanced accuracy scores range from 0.53 to 0.93 with model 6 yielding the highest score. Looking at Table 1, the F1 score has obviously improved for model 5 and 6. However, it is still very low indicating that all six models were potentially not performing well when dealing with the high-risk loan prediction. Looking at the precision and recall scores, the low values for F1 are mainly due to the very low precision scores indicating a large number of false-positive outputs. On the other hand, the recall scores are relatively high and range from 0.60 to 0.91. The high recall score indicates a low number of false-negative output that might be more important than the precision (and F1) score when dealing with the high-risk loan prediction. For instance, a recall score of 0.91 for model 6 means that only 9% of the high-risk loans were falsely predicted as low-risk loans. For the same model, the precision score is 0.07 which is pretty low, but it might not be very consequential since this indicates a large number of false-positive outputs (low-risk loans predicted as high-risk loans) which might get resolved by more in-depth evaluation of the applications by the company. Therefore, model 6 might be considered as a reasonable model when it comes to predicting high-risk loan applications (recall score is very high and low F1 and precision scores might not matter much). Looking at Table 2, the F1 score range from 0.62 to 0.97 with model 5 and 6 having the highest values. All six models have performed relatively well when dealing with the low-risk loan predictions. Overall, model 6 has done the best job predicting the low-risk loan applications.
  
 <div align="center">    
  
Table 1: Model performance for high-risk loans
   
![image](https://user-images.githubusercontent.com/103223944/183335574-123b2ebc-3022-4588-b437-ef64043dbc53.png)
   
Table 2: Model performance for low-risk loans
   
![image](https://user-images.githubusercontent.com/103223944/183335617-ec8acc91-202a-45b1-85e4-ffe65bb67655.png)
