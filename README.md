# Grammar-Scoring-Engine

## Competition in Kaggle

## Problem to be solved 
- automated Grammar score based on spoken audio files
- teach  model how to evaluate English speech

Given:
- Training data = 769 audio files (.wav) + train.csv [filename, label]
- Test data = 216 files (.wav)

Evaluation 
- on how preprocess the audio data
- select an appropriate methodology to solve the problem
- evaluate its performance using relevant metrics.
- Jupyter Notebook with well-documented and commented code.
- Brief report within the same notebook that explains
    - the approach, preprocessing steps, pipeline architecture, and evaluation results.
- compulsory to add rmse score of the training data in final submission notebook.
- Leaderboard evaluation @ Pearson Correlation and RMSE
- Add proper visualization
  
Evaluation Criteria
1.	Correctness: Does the solution work as expected?
2.	Code Quality: Is the code clean, well-structured, and documented?
3.	Performance: How well does the model perform on the test dataset?
4.	Interpretability: Are the results well-explained with relevant visualizations?

Note - compute task relevant metrics in your notebook to benchmark model performance. You have to add visualizations wherever applicable.
-----
Load datasets

        import pandas as pd
        
        train_df = pd.read_csv('/kaggle/input/shl-hiring-assessment-2026/train.csv')
        
        test_df = pd.read_csv('/kaggle/input/shl-hiring-assessment-2026/test.csv')

## EXPLORATION + inference

- Datasets do not have null values
- duration has only a weak relationship with the score Pearson correlation = 0.087
- scores range from 0 to 5 and have half point values - Use Regression
- Baseline RMSE = 1.36

## TRIAL 1

**Continuous values = Regression will be used**

### 31 acoustic features extracted

**Training - Random Forest + 31 Acoustic Features**

Random Forest RMSE  = 0.74
Random Forest Pearson = 0.84
Training RMSE: 0.2987

***Trial 1 Public Score: 0.7494***

## Trial 2 

**Improving audio based approach**

**Train @ ExtraTreesRegressor**
Validation RMSE: 0.6813
Validation Pearson: 0.8662

Hence, better than Random Forest.
But Training RMSE = 0.0 = **Overfitting**
Check with less complex model of Extra Trees - RMSE increases
Hence, keep earlier Extra Trees model

### Cross Validation

RMSE = 0.76
Pearson = 0.79

hence = Extra trees is not kept
go back to Random forest model of trial 1






