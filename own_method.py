import numpy as np 
import pickle
import collections
import time 

###  Load useful data 
id2doc_name = open('parsed_data/id2doc_name.pickle', 'rb')
id2doc_name = pickle.load(id2doc_name)

valid_doc_tokens = open('parsed_data/valid_doc_tokens.pickle', 'rb')
valid_doc_tokens = pickle.load(valid_doc_tokens)

valid_query_tokens = open('parsed_data/valid_query_tokens.pickle', 'rb')
valid_query_tokens = pickle.load(valid_query_tokens)

id2word = open('parsed_data/id2word.pickle', 'rb')
id2word = pickle.load(id2word)

word2id = open('parsed_data/word2id.pickle', 'rb')
word2id = pickle.load(word2id)

doc_dict = open('parsed_data/doc_dict.pickle', 'rb')
doc_dict = pickle.load(doc_dict)

query_dict = open('parsed_data/query_dict.pickle', 'rb')
query_dict = pickle.load(query_dict)

inverted_index = open('parsed_data/inverted_index.pickle', 'rb')
inverted_index = pickle.load(inverted_index)


start_time = time.time()

num_documents = len(doc_dict)

up_weights = 3

a = open('raw_data/ohsumed.88-91','rb')
a = a.readlines()
doc_title=[]
for i,j in enumerate(a):
    if a[i].strip().decode('utf-8')=='.U':
        doc_id = a[i+1].strip().decode('utf-8')
        if a[i+6].strip().decode('utf-8')=='.T':
            doc_t = a[i+7].strip().decode('utf-8')
            doc_title.append(doc_t)


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import snowball
stemmer = snowball.SnowballStemmer('english')
from nltk.corpus import stopwords
stwords = set(stopwords.words('english'))

doc_title_dict = collections.defaultdict(list)
for i in range(num_documents):
    if i%1000==0:
        print(i)
    title = doc_title[i]
    title_token = tokenizer.tokenize(title)
    for token in title_token:
        token_lower = token.lower()
        if token_lower not in stwords:
            token_stem = stemmer.stem(token_lower)
            doc_title_dict[i].append(token_stem)

for i in range(num_documents):
    doc_dict[i] = doc_dict[i] + doc_title_dict[i]*up_weights

document_lengths = []
for i in range(num_documents):
	document_lengths.append(len(doc_dict[i]))

average_document_length = np.mean(document_lengths)

valid_query_tokens_id = []
for i in valid_query_tokens:
    valid_query_tokens_id.append(word2id[i])


inverted_index = collections.defaultdict(dict)
for doc_id in range(num_documents):
    doc_cont = doc_dict[doc_id]
    word_counter = collections.Counter(doc_cont)
    for query_id in valid_query_tokens_id:
        term_frenq = word_counter.get(query_id, 0)
        if term_frenq == 0:
            continue
        inverted_index[query_id][doc_id] = term_frenq



topk = 50
algo_name = 'own_method'

def bm25(document_id, query_name):
    k1=1.2
    b=0.75
    score = 0
    query_terms = query_dict[query_name]
    for query_term in query_terms:
        if document_id in inverted_index[query_term]:
            tf = inverted_index[query_term][document_id]
            Ld = document_lengths[document_id]
            df = len(inverted_index[query_term])
            idf = num_documents/df
            score += np.log(idf) * ((k1+1)*tf)/(k1*((1-b)+b*(Ld/average_document_length))+tf)
    return score




results_store = collections.defaultdict(list)


for query_name in query_dict:
	for doc_id in range(num_documents):
		score = bm25(doc_id,query_name)
		results_store[query_name].append([score,id2doc_name[doc_id]])

with open('results/own_method_results.out', 'w') as outputfile:
	for query_name in results_store:
		sorted_doc_store = sorted(results_store[query_name],reverse=True)[:topk]
		for i,j in enumerate(sorted_doc_store):
			outputfile.write(query_name + ' ' + 'Q0' + ' '+ j[1] + ' ' + str(i+1) + ' ' + str(j[0]) + ' ' + algo_name + '\n')


seconds = time.time() - start_time
print('running own method algorithm using',seconds, 'seconds.')


