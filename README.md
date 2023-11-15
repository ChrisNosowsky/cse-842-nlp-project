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
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.33    |  0.09  |   0.14   |  1900   |
| Sci/Tech                  |   0.18    |  0.60  |   0.28   |  1900   |
| Sports                    |   0.15    |  0.25  |   0.19   |  1900   |
| World                     |   0.19    |  0.10  |   0.13   |  1900   |
| alt.atheism               |   0.10    |  0.01  |   0.02   |   319   |
| comp.graphics             |   0.02    |  0.00  |   0.00   |   389   |
| comp.os.ms-windows.misc   |   0.00    |  0.00  |   0.00   |   394   |
| comp.sys.ibm.pc.hardware  |   0.00    |  0.00  |   0.00   |   392   |
| comp.sys.mac.hardware     |   0.00    |  0.00  |   0.00   |   385   |
| comp.windows.x            |   0.12    |  0.01  |   0.02   |   395   |
| misc.forsale              |   0.03    |  0.01  |   0.02   |   390   |
| rec.autos                 |   0.05    |  0.15  |   0.07   |   396   |
| rec.motorcycles           |   0.02    |  0.00  |   0.00   |   398   |
| rec.sport.baseball        |   0.05    |  0.04  |   0.04   |   397   |
| rec.sport.hockey          |   0.00    |  0.00  |   0.00   |   399   |
| sci.crypt                 |   0.00    |  0.00  |   0.00   |   396   |
| sci.electronics           |   0.00    |  0.00  |   0.00   |   393   |
| sci.med                   |   0.04    |  0.07  |   0.06   |   396   |
| sci.space                 |   0.04    |  0.00  |   0.00   |   394   |
| soc.religion.christian    |   0.08    |  0.09  |   0.09   |   398   |
| talk.politics.guns        |   0.00    |  0.00  |   0.00   |   364   |
| talk.politics.mideast     |   0.10    |  0.05  |   0.06   |   376   |
| talk.politics.misc        |   0.04    |  0.13  |   0.07   |   310   |
| talk.religion.misc        |   0.00    |  0.00  |   0.00   |   251   |

|        Accuracy           |           |           |           |  0.14   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.06    |   0.07    |   0.05    |         |
|   Weighted Avg            |   0.12    |   0.14    |   0.10    |         |

#### NGRAMS
TBD
#### DOC2VEC
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.00    |  0.00  |   0.00   |  1900   |
| Sci/Tech                  |   0.12    |  0.24  |   0.16   |  1900   |
| Sports                    |   0.00    |  0.00  |   0.00   |  1900   |
| World                     |   0.25    |  0.81  |   0.39   |  1900   |
| alt.atheism               |   0.00    |  0.00  |   0.00   |   319   |
| comp.graphics             |   0.00    |  0.00  |   0.00   |   389   |
| comp.os.ms-windows.misc   |   0.00    |  0.00  |   0.00   |   394   |
| comp.sys.ibm.pc.hardware  |   0.06    |  0.07  |   0.06   |   392   |
| comp.sys.mac.hardware     |   0.00    |  0.00  |   0.00   |   385   |
| comp.windows.x            |   0.07    |  0.64  |   0.13   |   395   |
| misc.forsale              |   0.07    |  0.15  |   0.09   |   390   |
| rec.autos                 |   0.00    |  0.00  |   0.00   |   396   |
| rec.motorcycles           |   0.00    |  0.00  |   0.00   |   398   |
| rec.sport.baseball        |   0.05    |  0.07  |   0.06   |   397   |
| rec.sport.hockey          |   0.00    |  0.00  |   0.00   |   399   |
| sci.crypt                 |   0.00    |  0.00  |   0.00   |   396   |
| sci.electronics           |   0.00    |  0.00  |   0.00   |   393   |
| sci.med                   |   0.00    |  0.00  |   0.00   |   396   |
| sci.space                 |   0.00    |  0.00  |   0.00   |   394   |
| soc.religion.christian    |   0.00    |  0.00  |   0.00   |   398   |
| talk.politics.guns        |   0.00    |  0.00  |   0.00   |   364   |
| talk.politics.mideast     |   0.00    |  0.00  |   0.00   |   376   |
| talk.politics.misc        |   0.00    |  0.00  |   0.00   |   310   |
| talk.religion.misc        |   0.00    |  0.00  |   0.00   |   251   |

|        Accuracy           |           |           |           |  0.16   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.03    |   0.08    |   0.04    |         |
|   Weighted Avg            |   0.05    |   0.16    |   0.08    |         |
#### WORD2VEC
TBD

### RIPPER Model
#### BOW
TBD
#### TFIDF
TBD
#### NGRAMS
TBD
#### DOC2VEC
TBD
#### WORD2VEC
TBD

### Naive Bayes Model
#### BOW
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.85    |  0.80  |   0.82   |  1900   |
| Sci/Tech                  |   0.84    |  0.76  |   0.80   |  1900   |
| Sports                    |   0.94    |  0.93  |   0.93   |  1900   |
| World                     |   0.88    |  0.87  |   0.87   |  1900   |
| alt.atheism               |   0.78    |  0.82  |   0.80   |   319   |
| comp.graphics             |   0.53    |  0.79  |   0.64   |   389   |
| comp.os.ms-windows.misc   |   0.33    |  0.01  |   0.01   |   394   |
| comp.sys.ibm.pc.hardware  |   0.52    |  0.74  |   0.61   |   392   |
| comp.sys.mac.hardware     |   0.69    |  0.83  |   0.75   |   385   |
| comp.windows.x            |   0.81    |  0.70  |   0.75   |   395   |
| misc.forsale              |   0.72    |  0.82  |   0.77   |   390   |
| rec.autos                 |   0.82    |  0.87  |   0.84   |   396   |
| rec.motorcycles           |   0.85    |  0.95  |   0.90   |   398   |
| rec.sport.baseball        |   0.86    |  0.92  |   0.89   |   397   |
| rec.sport.hockey          |   0.92    |  0.97  |   0.94   |   399   |
| sci.crypt                 |   0.88    |  0.91  |   0.89   |   396   |
| sci.electronics           |   0.72    |  0.71  |   0.71   |   393   |
| sci.med                   |   0.80    |  0.81  |   0.81   |   396   |
| sci.space                 |   0.75    |  0.89  |   0.81   |   394   |
| soc.religion.christian    |   0.86    |  0.92  |   0.89   |   398   |
| talk.politics.guns        |   0.74    |  0.90  |   0.81   |   364   |
| talk.politics.mideast     |   0.90    |  0.85  |   0.88   |   376   |
| talk.politics.misc        |   0.60    |  0.62  |   0.61   |   310   |
| talk.religion.misc        |   0.63    |  0.62  |   0.63   |   251   |

|        Accuracy           |           |           |           |  0.81   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.76    |   0.79    |   0.77    |         |
|   Weighted Avg            |   0.81    |   0.81    |   0.80    |         |
#### TFIDF
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.16    |  0.03  |   0.05   |  1900   |
| Sci/Tech                  |   0.25    |  0.15  |   0.19   |  1900   |
| Sports                    |   0.37    |  0.11  |   0.17   |  1900   |
| World                     |   0.41    |  0.14  |   0.21   |  1900   |
| alt.atheism               |   0.06    |  0.09  |   0.07   |   319   |
| comp.graphics             |   0.05    |  0.08  |   0.06   |   389   |
| comp.os.ms-windows.misc   |   0.03    |  0.05  |   0.03   |   394   |
| comp.sys.ibm.pc.hardware  |   0.04    |  0.03  |   0.04   |   392   |
| comp.sys.mac.hardware     |   0.10    |  0.11  |   0.10   |   385   |
| comp.windows.x            |   0.10    |  0.12  |   0.11   |   395   |
| misc.forsale              |   0.03    |  0.06  |   0.04   |   390   |
| rec.autos                 |   0.02    |  0.02  |   0.02   |   396   |
| rec.motorcycles           |   0.02    |  0.02  |   0.02   |   398   |
| rec.sport.baseball        |   0.05    |  0.06  |   0.05   |   397   |
| rec.sport.hockey          |   0.05    |  0.03  |   0.04   |   399   |
| sci.crypt                 |   0.03    |  0.03  |   0.03   |   396   |
| sci.electronics           |   0.07    |  0.15  |   0.09   |   393   |
| sci.med                   |   0.04    |  0.08  |   0.05   |   396   |
| sci.space                 |   0.05    |  0.10  |   0.07   |   394   |
| soc.religion.christian    |   0.07    |  0.05  |   0.06   |   398   |
| talk.politics.guns        |   0.04    |  0.06  |   0.05   |   364   |
| talk.politics.mideast     |   0.08    |  0.11  |   0.09   |   376   |
| talk.politics.misc        |   0.06    |  0.24  |   0.10   |   310   |
| talk.religion.misc        |   0.05    |  0.29  |   0.08   |   251   |

|        Accuracy           |           |           |           |  0.10   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.09    |   0.09    |   0.08    |         |
|   Weighted Avg            |   0.17    |   0.10    |   0.11    |         |
#### NGRAMS
TBD
#### DOC2VEC
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.20    |  0.42  |   0.27   |  1900   |
| Sci/Tech                  |   0.17    |  0.59  |   0.27   |  1900   |
| Sports                    |   0.17    |  0.29  |   0.22   |  1900   |
| World                     |   0.25    |  0.19  |   0.22   |  1900   |
| alt.atheism               |   0.00    |  0.00  |   0.00   |   319   |
| comp.graphics             |   0.00    |  0.00  |   0.00   |   389   |
| comp.os.ms-windows.misc   |   0.00    |  0.00  |   0.00   |   394   |
| comp.sys.ibm.pc.hardware  |   0.00    |  0.00  |   0.00   |   392   |
| comp.sys.mac.hardware     |   0.00    |  0.00  |   0.00   |   385   |
| comp.windows.x            |   0.00    |  0.00  |   0.00   |   395   |
| misc.forsale              |   0.00    |  0.00  |   0.00   |   390   |
| rec.autos                 |   0.00    |  0.00  |   0.00   |   396   |
| rec.motorcycles           |   0.00    |  0.00  |   0.00   |   398   |
| rec.sport.baseball        |   0.00    |  0.00  |   0.00   |   397   |
| rec.sport.hockey          |   0.00    |  0.00  |   0.00   |   399   |
| sci.crypt                 |   0.00    |  0.00  |   0.00   |   396   |
| sci.electronics           |   0.00    |  0.00  |   0.00   |   393   |
| sci.med                   |   0.00    |  0.00  |   0.00   |   396   |
| sci.space                 |   0.00    |  0.00  |   0.00   |   394   |
| soc.religion.christian    |   0.00    |  0.00  |   0.00   |   398   |
| talk.politics.guns        |   0.00    |  0.00  |   0.00   |   364   |
| talk.politics.mideast     |   0.00    |  0.00  |   0.00   |   376   |
| talk.politics.misc        |   0.00    |  0.00  |   0.00   |   310   |
| talk.religion.misc        |   0.00    |  0.00  |   0.00   |   251   |

|        Accuracy           |           |           |           |  0.19   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.03    |   0.06    |   0.04    |         |
|   Weighted Avg            |   0.10    |   0.19    |   0.12    |         |
#### WORD2VEC
TBD

### BERT Model
#### BOW
TBD
#### TFIDF
TBD
#### NGRAMS
TBD
#### DOC2VEC
TBD
#### WORD2VEC
TBD

## Result Notes (Dev Notes -- Add here)
Stemmers improved model



## Disclosures
Pre-trained GoogleNews vector for word embeddings downloaded from here:
https://github.com/mmihaltz/word2vec-GoogleNews-vectors

Chris uses an AMD GPU -- had to download tensorflow-directml-plugin for AMD GPU support.