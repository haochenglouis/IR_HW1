
# Search Engine for Information Retrieval

This repository is the implementation of the search engine for Homework 1 in the Information Retrieval Class (CSE272 Spring UCSC). All the code are written in python.

## Prerequisites

Python 3

[nltk toolkit](https://www.nltk.org) (pip install nltk)

[trec_eval](https://github.com/usnistgov/trec_eval) (cd trec_eval; make)


## Guideline

### Downloading dataset: 

Download the dataset from "[google drive](https://drive.google.com/drive/folders/1fhrFtDtDWsxRJ2ND0zapsiZJUwBU7-4a?usp=sharing)" and put all the file in **raw_data/**. In the raw data, "ohsumed.88-91" is the collection of the documents; "query.ohsu.88-91" is the query file; "qrels.ohsu.1-63" is the relevance file.


### Run data parser to generate inverted index

Run command below:

```
python data_parser.py
```

This command will parse the documents and querys to generate inverted index and other useful data. These data are stored in **parsed_data/**. This command will also parse the relevance file to make it friendly for  [trec_eval](https://github.com/usnistgov/trec_eval) [Simply add "0" in each line]

### Run each algorithm to get ranking results:

In the repository, there are "boolean.py","tf.py", "tfidf.py", etc, indicating each method. To get the ranking results, just directly run the python file. For example, for tf-idf method, just run the command below:

```
python tfidf.py
```

The above command will put the ranking results in **results/**

### Trec evaluation

Suppose trec folder and this folder are in the same repository. To get the evaluation results of tf-idf, simply run: 

```
./trec_eval -m all_trec -q ../IR_HW1/raw_data/qrels.ohsu.88-91_trec_friendly ../IR_HW1/results/tfidf_results.out > ../IR_HW1_ori/results/tfidf_evaluation.out 
```

'tfidf_evaluation.out' will contain many different evaluation measure results. 

### Plot figure

After running all the algorithms in the repository including boolean, tf, tf-idf, relevance feedback, own method. Simply run

```
python plot_results.py
```

Note you may need to modify the path and file name to make it run successfully. This command will generate two figures in **results/**. You can compare the performance of each algorithm by viewing these two figures.