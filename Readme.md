## Overview

This project implements Yarowsky’s (1995) bootstrapping algorithm for Word Sense Disambiguation (WSD) using the AP News 1988 corpus.
The implementation includes:

	1.	A manual evaluation pipeline on naturally ambiguous English words.
	2.	An auto evaluation pipeline using conflated nouns.
	3.	Bootstrapped decision list training using seed cues.
	4.	One Sense Per Discourse (OSPD) post-processing.

The codebase allows running each pipeline independently.

## Description

### 1. Preprocessing:

Extract TEXT blocks, split into sentences, tokenize, and build positional index.

- python3 -m src.preprocess

produces,

    - data/preprocessed/corpus.pkl
    - data/preprocessed/token_index.pkl

### 2. Manual Evaluation:
#### 2.1 Build Feature sets:

- python3 -m src.feature_selection

produces,

    - feature_sets.pkl

#### 2.2 Train Decision Lists:

- python3 -m src.decision_list

produces,

    -  decision_lists.pkl

#### 2.3 Generate Manual Evaluation CSV

- python3 -m src.manual_evaluation

produces,

    - data/output/manual_evaluation_samples.csv

This gold_label of the csv requires to be filled manually.

#### 2.4 Compute Manual Accuracy

- python3 -m src.manual_accuracy

Prints accuracy per word

### 3. Auto Evaluation: 

Synthetic polysemous tokens are formed by replacing nouns with conflated pairs (e.g., car + speech → carspeech).
Gold labels come from the original source word.

#### 3.1 Build Synthetic Corpus + Features

- python3 -m src.synthetic_wsd

produces,
	- synthetic_corpus.pkl
	- feature_sets.pkl
	- gold_labels.pkl
	- targets.pkl

#### 3.2 Train Decision Lists on Synthetic Data

- python3 -m src.synthetic_train

produces,

    - decision_lists.pkl

#### 3.3 Automatic Evaluation

Enable or disable OSPD toggle inside the script by setting it to True or False respectively.

- python3 -m src.synthetic_eval

Outputs accuracy for each conflated word pair.

### 4. One Sense Per Discourse Implementation

OSPD is implemented in src/synthetic_ospd.py.

- python3 -m src.synthetic_eval

It applies after prediction but before computing accuracy:

	-   Group predictions by each document.
	-	Compute majority sense.
	-	Enforce majority only if agreement >= threshold.
	-	Otherwise reject predictions (set to None).

Threshold is adjustable (currently set to 0.55).


### Code Execution:

#### Manual evaluation:

Execute the following in the below order,

1. python3 -m src.preprocess
2. python3 -m src.feature_selection
3. python3 -m src.core.decision_list
4. python3 -m src.manual_evaluation
5. python3 -m src.manual_accuracy


#### Auto evaluation:

Execute the following in the below order,

1. python3 -m src.synthetic_wsd
2. python3 -m src.synthetic_train
3. Edit USE_OSPD toggle in synthetic_eval.py
4. python3 -m src.synthetic_eval