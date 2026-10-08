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

Train with ExtraTreesRegressor

Validation RMSE = 0.6813
Validation Pearson = 0.8662

Lower RMSE = better. Higher Pearson = better. Hence, better than Random Forest.

But training RMSE = 0.0. Training RMSE = 0 indicates the model fits the training data almost perfectly, suggesting possible overfitting.

Check with a less complex Extra Trees model: RMSE increases (0.7014).

The less-complex Extra Trees model had higher RMSE, so the original Extra Trees model performed better on this validation split.
Cross Validation
RMSE = 0.76
Pearson = 0.79

**Extra Trees performed better than Random Forest on the initial validation split. However, its training RMSE of 0 indicated possible overfitting. A less-complex Extra Trees model performed worse. Cross-validation was then used to check whether performance was consistent. It achieved RMSE = 0.76 and Pearson = 0.79. Based on these results, the original Extra Trees model was kept.**

### Trial 2 public score: 0.7455 (only a small gain over Random Forest, so the audio features had reached a ceiling)

## Why the audio-only approach plateaued

MFCC, ZCR and RMS describe how the audio sounds, not what is said. Grammar lives in the words. So the next trials add features computed from the text of the speech.

# TRIAL 3: Whisper transcripts + simple text features

## Step 1: Transcribe all 985 audio files with Whisper ("small")

Whisper is an external model, allowed by the rules.
Transcripts were saved to CSV (train_text.csv, test_text.csv) so the slow step is never repeated.
5 transcripts were empty. All 5 have label 0.0, so "no speech" means score 0.

## Step 2: 10 simple text features n_words, n_unique, unique_ratio, avg_word_len, n_sentences, avg_sent_len, n_fillers (um, uh), n_repeats, n_commas, words_per_sec.

Why: longer, more varied, smoother speech usually scores higher.

5-fold cross-validation (Extra Trees)

Features	CV RMSE	CV Pearson
Audio only	0.7607	0.7911
Audio + text counts	0.7243	0.8122

Random Forest was re-tested on the new features: CV RMSE 0.7457, Pearson 0.7987. Extra Trees stayed better.

## Trial 3 public score: 0.6077

Submission file check: sample_submission.csv has 204 rows, test.csv has 216 (all matching the audio folder). The submission is built from test.csv. Predictions are clipped to 0-5.

## TRIAL 4: Sentence embeddings (MiniLM) + PCA
all-MiniLM-L6-v2 turns each transcript into 384 numbers that capture meaning and style.
384 extra columns confuse Extra Trees (only 769 training files), so PCA shrinks them to 20.
Features	CV RMSE	CV Pearson
Current best	0.7243	0.8122
+ all 384 embedding numbers	0.7394	0.8053
+ 20 PCA numbers	0.7090	0.8220

Is the gain real or luck? Re-tested with 3 different random splits:

Seed	Without embeddings	With PCA 20
1	0.7234	0.7056
2	0.7138	0.6993
3	0.7146	0.7041

PCA 20 won every time, so it was kept.

Trial 4 public score: 0.5706

## TRIAL 5: Blend of different models
Extra Trees (audio + text + PCA 20) + Ridge (alpha = 300) + SVR (C = 1).

Ridge and SVR use StandardScaler and all 384 embedding numbers.

Final prediction = simple average of the three.

Why: trees and linear models make different kinds of mistakes, so averaging often beats any single model.

Seed	Extra Trees alone	Blend of 3
1	0.7056	0.6990
2	0.6993	0.6932
3	0.7041	0.6913

Ridge (about 0.73) and SVR (about 0.75) alone are worse than Extra Trees, so they are useful only as partners.

Trial 5 public score: 0.5434

TRIAL 6: GPT-2 fluency features
For each transcript, measure how "surprised" GPT-2 is by the text: nll_mean, nll_max, nll_std.
Why: text with grammar errors usually surprises a language model more, so this is a direct fluency signal.
Seed	ET (before)	ET + GPT-2

1	0.7056	0.6897
2	0.6993	0.6844
3	0.7041	0.6915

GPT-2 features were added to Extra Trees and to Ridge/SVR, then the 3-model blend was rebuilt.

## Trial 6 public score: 0.5237

TRIAL 7: CoLA grammar-acceptability features
Model: textattack/roberta-base-CoLA. It is trained to answer "is this sentence grammatical?".
Each transcript is split into sentences. Each sentence gets a probability of being grammatical (LABEL_1).
Features per file: cola_mean (average), cola_min (worst sentence), cola_bad (share of sentences below 0.5).
Sanity check: "I love the sound of the waves." scored 0.98; "...seashells that has been washed..." scored 0.31.
Seed	ET + GPT-2	ET + GPT-2 + CoLA	Blend (final)
1	0.6897	0.6679	0.6604
2	0.6844	0.6699	0.6559
3	0.6915	0.6708	0.6566

## Trial 7 public score: 0.5029 (final)

Final pipeline architecture
Audio (.wav)
   |-- librosa --------------------> 31 acoustic features
   |-- Whisper (small) -> transcript
          |-- counting rules -------> 10 text features
          |-- MiniLM embedding -----> 384 numbers -> PCA 20
          |-- GPT-2 ----------------> 3 fluency features
          |-- CoLA (per sentence) --> 3 grammar features

Extra Trees  (audio + text + PCA 20 + GPT-2 + CoLA)  --\
Ridge        (audio + text + 384 emb + GPT-2 + CoLA) ---|--> average --> clip to 0-5
SVR          (audio + text + 384 emb + GPT-2 + CoLA) --/
Score history (public leaderboard, lower is better)
Trial	What changed	Public score
1	Random Forest + 31 audio features	0.7494
2	Extra Trees + 31 audio features	0.7455
3	+ Whisper text features	0.6077
4	+ MiniLM embeddings (PCA 20)	0.5706
5	+ Ridge/SVR blend	0.5434
6	+ GPT-2 fluency	0.5237
7	+ CoLA grammar	0.5029


***How every idea was tested***


5-fold cross-validation: the model is always scored on data it did not train on.
Three different random splits (seeds 1, 2, 3): one split can be lucky.
A change is kept only if it wins in all 3 seeds.
Every submission is saved under a new name, so a good file is never overwritten.
Training RMSE of the final model (compulsory)

TODO: add the number here after running the final model on the training data. Expect a low value because trees can memorize training data (see the overfitting note in Trial 2). Cross-validation RMSE (about 0.66) is the honest estimate of performance on unseen data.

# What did not help
All 384 embedding numbers in Extra Trees (worse than PCA 20)
Less complex (tuned) Extra Trees
Random Forest (worse than Extra Trees, on audio features and on the final features)
# Limitations
Whisper can silently "correct" grammar mistakes, which hides the errors we want to detect.
Grammar scores are partly subjective, so there is probably a floor on how low the error can go.
Cross-validation RMSE (about 0.66) is higher than the public score (0.50), so the public test set appears easier than the training folds. Treat the public score as a rough guide.
The training set is small (769 files), so small CV gains can be noise. This is why the 3-seed rule is used.
# Future ideas
Better embedding model (all-mpnet-base-v2) and PCA 10 or 30
Whisper confidence and pause features
Audio embeddings (wav2vec2 or the Whisper encoder)
Tuned blend weights
Tools used

Python, pandas, numpy, scipy, scikit-learn, librosa, openai-whisper, sentence-transformers, transformers (GPT-2, CoLA). Run on Kaggle with GPU (T4) and Internet on for model downloads.

# How to reproduce
Create a Kaggle notebook, add the competition data, turn on GPU and Internet.
Run the notebook in order: audio features, Whisper transcripts (slowest, about 15-40 min), text features, embeddings + PCA, GPT-2, CoLA, then the blend.
Download submission.csv and upload it on the competition's Submit Predictions page.
Data note

The competition data (audio, labels, and transcripts derived from them) is not included in this repository. Download it from the Kaggle competition page and follow its rules.
-------
# Summary
-------


Results
Step	What was added	Kaggle score (lower is better)
1	31 audio features + Extra Trees	0.7455
2	+ simple text features from Whisper transcripts	0.6077
3	+ sentence embeddings (MiniLM, shrunk with PCA to 20 numbers)	0.5706
4	+ blend of Extra Trees, Ridge and SVR	0.5434
5	+ GPT-2 fluency scores	0.5237
6	+ CoLA grammar-acceptability scores	0.5029








