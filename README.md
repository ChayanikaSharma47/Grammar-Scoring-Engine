# Grammar-Scoring-Engine

Final Kaggle score: 0.5029 (started at 0.7455)

This project was built step by step, testing every idea before keeping it.

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
        
        train_df = pd.read_csv('/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/train.csv')
        
        test_df = pd.read_csv('/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/test.csv')

## EXPLORATION + inference

- Datasets do not have null values
- duration has only a weak relationship with the score Pearson correlation = 0.087
- scores range from 0 to 5 and have half point values - Use Regression
- Baseline RMSE = 1.36

## TRIAL 1

**Continuous values = Regression will be used**

### 31 acoustic features extracted

**Random Forest + 31 Acoustic Features**

Validation RMSE  = 0.74
Validation Pearson = 0.84

***Trial 1 Public Score: 0.7494***

## Trial 2 

**Improving audio based approach**

**Train @ ExtraTreesRegressor**
Validation RMSE: 0.6813
Validation Pearson: 0.8662

Lower RMSE = better.
Higher Pearson = better.
Hence, better than Random Forest.


But Training RMSE = 0.0 
**Training RMSE = 0 indicates the model fits the training data almost perfectly, suggesting possible overfitting.**

Check with less complex model of Extra Trees - RMSE increases

### The less-complex Extra Trees model had higher RMSE, so the original Extra Trees model performed better on this validation split.

## Cross Validation

RMSE = 0.76
Pearson = 0.79

***Extra Trees performed better than Random Forest on the initial validation split. However, its training RMSE of 0 indicated possible overfitting. A less-complex Extra Trees model performed worse. Cross-validation was then used to check whether the performance was consistent. It achieved RMSE = 0.76 and Pearson = 0.79. Based on the more reliable cross-validation results, Random Forest appears to be better.***


-------
-------



Results
Step	What was added	Kaggle score (lower is better)
1	31 audio features + Extra Trees	0.7455
2	+ simple text features from Whisper transcripts	0.6077
3	+ sentence embeddings (MiniLM, shrunk with PCA to 20 numbers)	0.5706
4	+ blend of Extra Trees, Ridge and SVR	0.5434
5	+ GPT-2 fluency scores	0.5237
6	+ CoLA grammar-acceptability scores	0.5029








