import nltk
import numpy as np
import collections 
import pickle
import time 
####parse the document#####################################################
#
#	we put the title and content in a doc. Note some docs only have the title.
#
#############################################################################

start_time = time.time()

a = open('raw_data/ohsumed.88-91','rb')
a = a.readlines()
doc_id_str = [] ## the original name of the doc id
docs = []       ## store the content of each doc 


for i,j in enumerate(a):
    if a[i].strip().decode('utf-8')=='.U':
        doc_id = a[i+1].strip().decode('utf-8')
        if a[i+6].strip().decode('utf-8')=='.T':
            doc_t = a[i+7].strip().decode('utf-8')
            if a[i+10].strip().decode('utf-8')=='.W':
                doc_s = a[i+11].strip().decode('utf-8')
                doc_id_str.append(doc_id)
                docs.append(doc_t + ' ' +doc_s)
            else:
                doc_id_str.append(doc_id)
                docs.append(doc_t)



### Map the int number to original name of the doc id
num_documents = len(doc_id_str)
id2str = {}
for i in range(num_documents):
    id2str[i] = doc_id_str[i]



### Using nltk tool to eliminate stop words, put all word to lowercase, stem every word
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import snowball
stemmer = snowball.SnowballStemmer('english')
from nltk.corpus import stopwords
stwords = set(stopwords.words('english'))


valid_tokens = set()   ## count valid tokens in the collection
id_doc = collections.defaultdict(list) ## store the int doc id and its corresponding documents

for i in range(num_documents):
    if i%1000==0: ## check speed
        print(i)
    doc_content = docs[i]
    doc_content_token = tokenizer.tokenize(doc_content)
    for token in doc_content_token:
        token_lower = token.lower()
        if token_lower not in stwords:
            token_stem = stemmer.stem(token_lower)
            valid_tokens.add(token_stem)
            id_doc[i].append(token_stem)




valid_tokens = list(valid_tokens)
num_valid_tokens = len(valid_tokens)
print('there are total', num_valid_tokens, 'valid tokens in the collections')
id2word = {}  ## map int number to token 
word2id={}    ## mao token to int number
for i,j in enumerate(valid_tokens):
    id2word[i] = j
    word2id[j] = i

###  apply word2id to all the documents
for i in id_doc:
    tokens = id_doc[i]
    tokens_ids = []
    for j in tokens:
        tokens_ids.append(word2id[j])
    id_doc[i] = tokens_ids



####parse the query##########################################################
#
#Similar to docs, we put the title and description together for each query id
#
##############################################################################


b = open('raw_data/query.ohsu.1-63','rb')
b = b.readlines()

valid_query_tokens = set()
query_dict = collections.defaultdict(list) ## store the query id and its content.
query_name = []
query_cotent =[]

for i,j in enumerate(b):
    if b[i].strip().decode('utf-8') == '<top>':
        query_name.append(b[i+1].strip().decode('utf-8')[14:])
        query_cotent.append(b[i+2].strip().decode('utf-8')[8:] +' ' +  b[i+4].strip().decode('utf-8')   )

num_querys = len(query_name)
for i in range(num_querys):
    content = query_cotent[i]
    content_token = tokenizer.tokenize(content)
    for token in content_token:
        token_lower = token.lower()
        if token_lower not in stwords:
            token_stem = stemmer.stem(token_lower)
            if token_stem in valid_tokens:
                valid_query_tokens.add(token_stem)
                query_dict[query_name[i]].append(token_stem)

valid_query_tokens = list(valid_query_tokens)
valid_query_tokens_ids = []
for i in valid_query_tokens:
    valid_query_tokens_ids.append(word2id[i])
for i in query_dict:
    tokens = query_dict[i]
    tokens_ids = []
    for j in tokens:
        tokens_ids.append(word2id[j])
    query_dict[i] = tokens_ids

####inverted index ##########################################################
#
# Generate inverted index using valid query terms
#
##############################################################################

inverted_index = collections.defaultdict(dict)
for doc_id in range(num_documents):
    doc_cont = id_doc[doc_id]
    word_counter = collections.Counter(doc_cont)
    for query_id in valid_query_tokens_ids:
        term_frenq = word_counter.get(query_id, 0)
        if term_frenq == 0:
            continue
        inverted_index[query_id][doc_id] = term_frenq

seconds = time.time() - start_time

print('Generating inverted index using',seconds, 'seconds.')



####qrel file ######################################################################
#
# Parsing the qrel file to make it friendly for trec_eval evaluation by inserting '0'
#
####################################################################################


c = open('raw_data/qrels.ohsu.88-91','rb')
c = c.readlines()
with open('raw_data/qrels.ohsu.88-91_trec_friendly', 'w') as newfile:
    for jj in c:
        s = jj.strip().decode('utf-8').split('\t')
        newfile.write(s[0]+' ' + '0' + ' ' + s[1] + ' ' +s[2]+'\n')


####store data ##########################################################
#
# Store useful data for further usage of algorithms
#
##############################################################################


inverted_index_ = open('parsed_data/inverted_index.pickle', 'wb')
pickle.dump(inverted_index, inverted_index_)
inverted_index_.close()

id2doc_name_ = open('parsed_data/id2doc_name.pickle', 'wb')
pickle.dump(id2str, id2doc_name_)
id2doc_name_.close()

valid_doc_tokens_ = open('parsed_data/valid_doc_tokens.pickle', 'wb')
pickle.dump(valid_tokens, valid_doc_tokens_)
valid_doc_tokens_.close()

valid_query_tokens_ = open('parsed_data/valid_query_tokens.pickle', 'wb')
pickle.dump(valid_query_tokens, valid_query_tokens_)
valid_query_tokens_.close()

id2word_ = open('parsed_data/id2word.pickle', 'wb')
pickle.dump(id2word, id2word_)
id2word_.close()

word2id_ = open('parsed_data/word2id.pickle', 'wb')
pickle.dump(word2id, word2id_)
word2id_.close()

doc_dict_ = open('parsed_data/doc_dict.pickle', 'wb')
pickle.dump(id_doc, doc_dict_)
doc_dict_.close()

query_dict_ = open('parsed_data/query_dict.pickle', 'wb')
pickle.dump(query_dict, query_dict_)
query_dict_.close()






