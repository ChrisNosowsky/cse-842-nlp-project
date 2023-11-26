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
## Results
- Vocab Size: 15,000
- Stemmers: Yes
- Lemmers: No

Result details can be found in the results' directory. 
We have detailed results covering class accuracy, recall, precision, f1-score, 
along with the overall metrics for each respective feature and model. 
Additionally, we have these results for the AG News, 20 News and combined datasets.

Below is the tabulated results of the overall performance for each dataset against the models and features.

===TBD===


## Result Notes (Dev Notes -- Add here)
Stemmers improved model
15K vocab limit better than 10K + 5K vocabs for NaiveBayes
NaiveBayes alpha = 0.1 is the best

## Disclosures
Pre-trained GoogleNews vector for word embeddings downloaded from here:
https://github.com/mmihaltz/word2vec-GoogleNews-vectors

Chris uses an AMD GPU -- had to download tensorflow-directml-plugin for AMD GPU support.