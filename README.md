# CSE 842 NLP Project: Article Text Categorization

Program Created By: Yue Deng, Josh Erno, Christopher Nosowsky


## How to run program (Python files)
Install the libraries:
```commandline
pip install -r requirements.txt
```

Then run the main program via. command line

```commandline
python main.py
```

You can also run it in an IDE of your choosing.


## How to run BERT classifier
Run it in a Jupyter notebook. Open the bert_classifier.ipynb file.

Make sure you have the saved preprocess files in the data/preprocess folder. 

There should be three total files for each dataset you plan to run.

You can also try to run it on your local machine within the main.py (setting the model list to BERT_MODEL), 
but it may be extremely slow to train.

----------------------
## Results - November
- Dataset: AG_NEWS + 20_NEWS
- Vocab Size: 15,000
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

#### NGRAMS (UniGram)
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.84    |  0.77  |   0.80   |  1900   |
| Sci/Tech                  |   0.75    |  0.88  |   0.81   |  1900   |
| Sports                    |   0.93    |  0.93  |   0.93   |  1900   |
| World                     |   0.89    |  0.82  |   0.85   |  1900   |
| alt.atheism               |   0.82    |  0.61  |   0.70   |   319   |
| comp.graphics             |   0.48    |  0.77  |   0.59   |   389   |
| comp.os.ms-windows.misc   |   0.65    |  0.66  |   0.66   |   394   |
| comp.sys.ibm.pc.hardware  |   0.50    |  0.76  |   0.60   |   392   |
| comp.sys.mac.hardware     |   0.69    |  0.80  |   0.74   |   385   |
| comp.windows.x            |   0.95    |  0.49  |   0.65   |   395   |
| misc.forsale              |   0.78    |  0.73  |   0.75   |   390   |
| rec.autos                 |   0.85    |  0.73  |   0.79   |   396   |
| rec.motorcycles           |   0.93    |  0.85  |   0.89   |   398   |
| rec.sport.baseball        |   0.82    |  0.92  |   0.86   |   397   |
| rec.sport.hockey          |   0.92    |  0.93  |   0.93   |   399   |
| sci.crypt                 |   0.93    |  0.84  |   0.88   |   396   |
| sci.electronics           |   0.66    |  0.54  |   0.59   |   393   |
| sci.med                   |   0.77    |  0.75  |   0.76   |   396   |
| sci.space                 |   0.90    |  0.82  |   0.86   |   394   |
| soc.religion.christian    |   0.80    |  0.83  |   0.82   |   398   |
| talk.politics.guns        |   0.70    |  0.84  |   0.76   |   364   |
| talk.politics.mideast     |   0.97    |  0.77  |   0.86   |   376   |
| talk.politics.misc        |   0.82    |  0.48  |   0.61   |   310   |
| talk.religion.misc        |   0.44    |  0.57  |   0.50   |   251   |

|        Accuracy           |           |           |           |  0.80   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.78    |   0.75    |   0.76    |         |
|   Weighted Avg            |   0.81    |   0.80    |   0.80    |         |

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
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.80    |  0.69  |   0.74   |  1900   |
| Sci/Tech                  |   0.78    |  0.70  |   0.74   |  1900   |
| Sports                    |   0.73    |  0.93  |   0.82   |  1900   |
| World                     |   0.83    |  0.75  |   0.79   |  1900   |
| alt.atheism               |   0.35    |  0.13  |   0.19   |   319   |
| comp.graphics             |   0.33    |  0.43  |   0.37   |   389   |
| comp.os.ms-windows.misc   |   0.55    |  0.18  |   0.27   |   394   |
| comp.sys.ibm.pc.hardware  |   0.27    |  0.40  |   0.32   |   392   |
| comp.sys.mac.hardware     |   0.27    |  0.48  |   0.35   |   385   |
| comp.windows.x            |   0.75    |  0.45  |   0.56   |   395   |
| misc.forsale              |   0.59    |  0.55  |   0.57   |   390   |
| rec.autos                 |   0.69    |  0.23  |   0.35   |   396   |
| rec.motorcycles           |   0.46    |  0.78  |   0.58   |   398   |
| rec.sport.baseball        |   0.57    |  0.43  |   0.49   |   397   |
| rec.sport.hockey          |   0.62    |  0.73  |   0.67   |   399   |
| sci.crypt                 |   0.81    |  0.82  |   0.82   |   396   |
| sci.electronics           |   0.45    |  0.29  |   0.36   |   393   |
| sci.med                   |   0.39    |  0.71  |   0.51   |   396   |
| sci.space                 |   0.74    |  0.49  |   0.59   |   394   |
| soc.religion.christian    |   0.41    |  0.93  |   0.57   |   398   |
| talk.politics.guns        |   0.52    |  0.68  |   0.59   |   364   |
| talk.politics.mideast     |   0.68    |  0.70  |   0.69   |   376   |
| talk.politics.misc        |   0.36    |  0.09  |   0.14   |   310   |
| talk.religion.misc        |   0.05    |  0.00  |   0.01   |   251   |

|        Accuracy           |           |           |           |  0.63   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.54    |   0.52    |   0.50    |         |
|   Weighted Avg            |   0.65    |   0.63    |   0.62    |         |


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
#### NGRAMS (UniGram)
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  |   0.84    |  0.80  |   0.82   |  1900   |
| Sci/Tech                  |   0.83    |  0.76  |   0.79   |  1900   |
| Sports                    |   0.94    |  0.93  |   0.93   |  1900   |
| World                     |   0.89    |  0.86  |   0.87   |  1900   |
| alt.atheism               |   0.77    |  0.81  |   0.79   |   319   |
| comp.graphics             |   0.53    |  0.78  |   0.63   |   389   |
| comp.os.ms-windows.misc   |   0.33    |  0.01  |   0.01   |   394   |
| comp.sys.ibm.pc.hardware  |   0.52    |  0.74  |   0.61   |   392   |
| comp.sys.mac.hardware     |   0.69    |  0.83  |   0.75   |   385   |
| comp.windows.x            |   0.81    |  0.70  |   0.75   |   395   |
| misc.forsale              |   0.71    |  0.83  |   0.77   |   390   |
| rec.autos                 |   0.82    |  0.87  |   0.85   |   396   |
| rec.motorcycles           |   0.84    |  0.95  |   0.89   |   398   |
| rec.sport.baseball        |   0.87    |  0.93  |   0.90   |   397   |
| rec.sport.hockey          |   0.93    |  0.97  |   0.95   |   399   |
| sci.crypt                 |   0.88    |  0.91  |   0.90   |   396   |
| sci.electronics           |   0.72    |  0.71  |   0.71   |   393   |
| sci.med                   |   0.80    |  0.82  |   0.81   |   396   |
| sci.space                 |   0.74    |  0.89  |   0.81   |   394   |
| soc.religion.christian    |   0.86    |  0.91  |   0.89   |   398   |
| talk.politics.guns        |   0.75    |  0.90  |   0.82   |   364   |
| talk.politics.mideast     |   0.92    |  0.85  |   0.88   |   376   |
| talk.politics.misc        |   0.61    |  0.62  |   0.61   |   310   |
| talk.religion.misc        |   0.63    |  0.62  |   0.63   |   251   |

|        Accuracy           |           |           |           |  0.81   |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.76    |   0.79    |   0.77    |         |
|   Weighted Avg            |   0.81    |   0.81    |   0.80    |         |

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
|                           | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Business                  | 0.76      | 0.67   | 0.72     |  1900   |
| Sci/Tech                  | 0.42      | 0.74   | 0.53     |  1900   |
| Sports                    | 0.73      | 0.89   | 0.80     |  1900   |
| World                     | 0.71      | 0.84   | 0.77     |  1900   |
| alt.atheism               | 0.62      | 0.26   | 0.37     |   319   |
| comp.graphics             | 0.35      | 0.46   | 0.40     |   389   |
| comp.os.ms-windows.misc   | 0.68      | 0.52   | 0.59     |   394   |
| comp.sys.ibm.pc.hardware  | 0.51      | 0.45   | 0.48     |   392   |
| comp.sys.mac.hardware     | 0.53      | 0.31   | 0.39     |   385   |
| comp.windows.x            | 0.71      | 0.51   | 0.59     |   395   |
| misc.forsale              | 0.42      | 0.62   | 0.50     |   390   |
| rec.autos                 | 0.53      | 0.35   | 0.42     |   396   |
| rec.motorcycles           | 0.47      | 0.60   | 0.53     |   398   |
| rec.sport.baseball        | 0.41      | 0.37   | 0.39     |   397   |
| rec.sport.hockey          | 0.83      | 0.26   | 0.40     |   399   |
| sci.crypt                 | 0.94      | 0.51   | 0.66     |   396   |
| sci.electronics           | 0.41      | 0.20   | 0.26     |   393   |
| sci.med                   | 0.39      | 0.25   | 0.30     |   396   |
| sci.space                 | 0.77      | 0.35   | 0.48     |   394   |
| soc.religion.christian    | 0.52      | 0.75   | 0.62     |   398   |
| talk.politics.guns        | 0.56      | 0.49   | 0.52     |   364   |
| talk.politics.mideast     | 0.93      | 0.31   | 0.46     |   376   |
| talk.politics.misc        | 0.80      | 0.15   | 0.25     |   310   |
| talk.religion.misc        | 0.50      | 0.01   | 0.02     |   251   |

|        Accuracy           |           |           |           |   0.59  |
|---------------------------|-----------|-----------|-----------|---------|
|      Macro Avg            |   0.60    |   0.45    |   0.48    |         |
|   Weighted Avg            |   0.62    |   0.59    |   0.57    |         |

### BERT Model
####  Features from BertTokenizer (since it is a pre-tained model)
|                          | Precision | Recall | F1-Score | Support |
|--------------------------|-----------|--------|----------|---------|
| Business                 | 0.86      | 0.85   | 0.86     | 1900    |
| Sci/Tech                 | 0.87      | 0.87   | 0.87     | 1900    |
| Sports                   | 0.96      | 0.98   | 0.97     | 1900    |
| World                    | 0.92      | 0.90   | 0.91     | 1900    |
| alt.atheism              | 0.74      | 0.69   | 0.71     | 319     |
| comp.graphics            | 0.76      | 0.73   | 0.75     | 389     |
| comp.os.ms-windows.misc  | 0.79      | 0.78   | 0.79     | 394     |
| comp.sys.ibm.pc.hardware | 0.66      | 0.73   | 0.69     | 392     |
| comp.sys.mac.hardware    | 0.80      | 0.81   | 0.80     | 385     |
| comp.windows.x           | 0.85      | 0.77   | 0.81     | 395     |
| misc.forsale             | 0.86      | 0.84   | 0.85     | 390     |
| rec.autos                | 0.90      | 0.88   | 0.89     | 396     |
| rec.motorcycles          | 0.92      | 0.84   | 0.88     | 398     |
| rec.sport.baseball       | 0.95      | 0.95   | 0.95     | 397     |
| rec.sport.hockey         | 0.96      | 0.97   | 0.97     | 399     |
| sci.crypt                | 0.92      | 0.88   | 0.90     | 396     |
| sci.electronics          | 0.75      | 0.83   | 0.79     | 393     |
| sci.med                  | 0.96      | 0.94   | 0.95     | 396     |
| sci.space                | 0.88      | 0.92   | 0.90     | 394     |
| soc.religion.christian   | 0.92      | 0.90   | 0.91     | 398     |
| talk.politics.guns       | 0.67      | 0.80   | 0.73     | 364     |
| talk.politics.mideast    | 0.97      | 0.90   | 0.93     | 376     |
| talk.politics.misc       | 0.60      | 0.56   | 0.58     | 310     |
| talk.religion.misc       | 0.61      | 0.69   | 0.65     | 251     |

|        Accuracy           |      |      |      | 0.86 |
|---------------------------|------|------|------|------|
|      Macro Avg            | 0.84 | 0.83 | 0.83 |      |
|   Weighted Avg            | 0.87 | 0.86 | 0.87 |      |

## Result Notes (Dev Notes -- Add here)
Stemmers improved model
15K vocab limit better than 10K + 5K vocabs for NaiveBayes
NaiveBayes alpha = 0.1 is the best

## Disclosures
Pre-trained GoogleNews vector for word embeddings downloaded from here:
https://github.com/mmihaltz/word2vec-GoogleNews-vectors

Chris uses an AMD GPU -- had to download tensorflow-directml-plugin for AMD GPU support.

Not pushing to repo the npz compressed files that get saved due to size (200MB each file)