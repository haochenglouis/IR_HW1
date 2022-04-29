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


num_documents = len(doc_dict)
document_lengths = []
for i in range(num_documents):
	document_lengths.append(len(doc_dict[i]))

average_document_length = np.mean(document_lengths)

topk = 50
algo_name = 'bm25'

def bm25(document_id, query_name):
    k1=1.2
    b=0.75
    score = 0
    query_terms = query_dict[query_name]
    for query_term in query_terms:
        if document_id in inverted_index[query_term]:
            tf = inverted_index[query_term][document_id]
            Ld = document_lengths[document_id]
            Lavg = average_document_length
            df = len(inverted_index[query_term])
            idf = num_documents/df
            first_term = np.log(idf)
            second_term = ((k1+1)*tf)/(k1*((1-b)+b*(Ld/Lavg))+tf)
            score += first_term * second_term 
    return score




results_store = collections.defaultdict(list)

start_time = time.time()
for query_name in query_dict:
	for doc_id in range(num_documents):
		score = bm25(doc_id,query_name)
		results_store[query_name].append([score,id2doc_name[doc_id]])

with open('results/bm25_results.out', 'w') as outputfile:
	for query_name in results_store:
		sorted_doc_store = sorted(results_store[query_name],reverse=True)[:topk]
		for i,j in enumerate(sorted_doc_store):
			outputfile.write(query_name + ' ' + 'Q0' + ' '+ j[1] + ' ' + str(i+1) + ' ' + str(j[0]) + ' ' + algo_name + '\n')


seconds = time.time() - start_time
print('running bm25 algorithm using',seconds, 'seconds.')


