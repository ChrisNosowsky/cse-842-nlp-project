# CSE 842 NLP Project: Article Text Categorization

Program Created By: Yue Deng, Josh Erno, Christopher Nosowsky


## How to run program
Install the libraries:
```commandline
pip install -r requirements.txt
```

Then run the main program via. command line

```commandline
python main.py
```

You can also run it in an IDE of your choosing.

----------------------
## Results - November
- Dataset: AG_NEWS + 20_NEWS
- Stemmers: Yes
- Lemmers: No

### Keras FCNN Model
#### BOW
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.89    |  0.73  |   0.80   |  1900   |
| Sci/Tech                  |   0.78    |  0.87  |   0.82   |  1900   |
| Sports                    |   0.90    |  0.96  |   0.93   |  1900   |
| World                     |   0.86    |  0.88  |   0.87   |  1900   |
| alt.atheism               |   0.71    |  0.67  |   0.69   |   319   |
| comp.graphics             |   0.53    |  0.75  |   0.62   |   389   |
| comp.os.ms-windows.misc   |   0.82    |  0.62  |   0.71   |   394   |
| comp.sys.ibm.pc.hardware  |   0.70    |  0.69  |   0.69   |   392   |
| comp.sys.mac.hardware     |   0.81    |  0.73  |   0.77   |   385   |
| comp.windows.x            |   0.84    |  0.69  |   0.76   |   395   |
| misc.forsale              |   0.74    |  0.82  |   0.78   |   390   |
| rec.autos                 |   0.86    |  0.84  |   0.85   |   396   |
| rec.motorcycles           |   0.88    |  0.93  |   0.90   |   398   |
| rec.sport.baseball        |   0.89    |  0.88  |   0.89   |   397   |
| rec.sport.hockey          |   0.93    |  0.95  |   0.94   |   399   |
| sci.crypt                 |   0.97    |  0.78  |   0.86   |   396   |
| sci.electronics           |   0.55    |  0.77  |   0.64   |   393   |
| sci.med                   |   0.81    |  0.76  |   0.78   |   396   |
| sci.space                 |   0.93    |  0.84  |   0.88   |   394   |
| soc.religion.christian    |   0.86    |  0.87  |   0.87   |   398   |
| talk.politics.guns        |   0.71    |  0.88  |   0.79   |   364   |
| talk.politics.mideast     |   0.97    |  0.76  |   0.86   |   376   |
| talk.politics.misc        |   0.81    |  0.57  |   0.67   |   310   |
| talk.religion.misc        |   0.48    |  0.63  |   0.54   |   251   |

|        Accuracy           |           |           |           |  0.82   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.80    |   0.79    |   0.79    |         |
|   Weighted Avg            |   0.83    |   0.82    |   0.82    |         |

#### TFIDF

#### NGRAMS
#### DOC2VEC

### RIPPER Model
#### BOW
#### TFIDF
#### NGRAMS
#### DOC2VEC

### Naive Bayes Model
#### BOW
#### TFIDF
#### NGRAMS
#### DOC2VEC

### BERT Model
#### BOW
#### TFIDF
#### NGRAMS
#### DOC2VEC


## Result Notes (Dev Notes -- Add here)
Stemmers improved model



## Disclosures
Pre-trained GoogleNews vector for word embeddings downloaded from here:
https://github.com/mmihaltz/word2vec-GoogleNews-vectors

Chris uses an AMD GPU -- had to download tensorflow-directml-plugin for AMD GPU support.