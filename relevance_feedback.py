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
document_lengths = []
for i in range(num_documents):
	document_lengths.append(len(doc_dict[i]))

average_document_length = np.mean(document_lengths)

selected_k = 20 #pseudo relevance of documents
selected_topquery = 20 #top k query terms in selected documents
new_inverted_index = collections.defaultdict(dict)

topk = 50
algo_name = 'relevance_feedback'

def tfidf(document_id, query_name,query_dict,inverted_index):
    score = 0
    query_terms = query_dict[query_name]
    for query_term in query_terms:
        if document_id in inverted_index[query_term]:
            tf = inverted_index[query_term][document_id]
            df = len(inverted_index[query_term])
            idf = num_documents/df
            score += np.log(1 + tf) * np.log(idf)
    return score

def bm25(document_id, query_name,query_dict,inverted_index):
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


valid_query_tokens_id = []
for i in valid_query_tokens:
	valid_query_tokens_id.append(word2id[i])

new_valid_query_tokens_id = set(valid_query_tokens_id)
new_query_dict = query_dict.copy()

results_store = collections.defaultdict(list)

for query_name in query_dict:
	print(query_name)
	for doc_id in range(num_documents):
		#score = tfidf(doc_id,query_name,query_dict,inverted_index)
		score = bm25(doc_id,query_name,query_dict,inverted_index)
		results_store[query_name].append([score,doc_id])
	sorted_doc_store = sorted(results_store[query_name],reverse=True)[:selected_k]
	doc_selected_all = []
	for i,j in sorted_doc_store:
		doc_selected_all = doc_selected_all + doc_dict[j]
	term_counter = collections.Counter(doc_selected_all).most_common(selected_topquery)
	for i,j in term_counter:
		new_valid_query_tokens_id.add(i)
		if i not in new_query_dict[query_name]:
			new_query_dict[query_name].append(i)

new_valid_query_tokens_id = list(new_valid_query_tokens_id)

print('there are', len(new_valid_query_tokens_id) - len(valid_query_tokens_id), 'new query terms')


##### generate new inverted index

for doc_id in range(num_documents):
    doc_cont = doc_dict[doc_id]
    word_counter = collections.Counter(doc_cont)
    for query_id in new_valid_query_tokens_id:
        term_frenq = word_counter.get(query_id, 0)
        if term_frenq == 0:
            continue
        new_inverted_index[query_id][doc_id] = term_frenq


results_store = collections.defaultdict(list)


for query_name in new_query_dict:
	print(query_name)
	for doc_id in range(num_documents):
		#score = tfidf(doc_id,query_name,new_query_dict,new_inverted_index)
		score = bm25(doc_id,query_name,new_query_dict,new_inverted_index)
		results_store[query_name].append([score,id2doc_name[doc_id]])

with open('results/relevance_feedback_results.out', 'w') as outputfile:
	for query_name in results_store:
		sorted_doc_store = sorted(results_store[query_name],reverse=True)[:topk]
		for i,j in enumerate(sorted_doc_store):
			outputfile.write(query_name + ' ' + 'Q0' + ' '+ j[1] + ' ' + str(i+1) + ' ' + str(j[0]) + ' ' + algo_name + '\n')


seconds = time.time() - start_time
print('running relevance feedback algorithm using',seconds, 'seconds.')


