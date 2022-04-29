import numpy as np 
import os 
import collections
import matplotlib.pyplot as plt


measure = ['P_5','P_10','P_30','P_100','recall_5','recall_10','recall_30','recall_100','map','ndcg']

boolean = open('results/boolean_evaluation.out','rb')
boolean = boolean.readlines()
boolean_eval = {}
for i in boolean:
	i = i.decode('utf-8').strip().split('\t')
	if i[0].strip() in measure:
		if i[1]=='all':
			boolean_eval[i[0].strip()] = float(i[2])

tf = open('results/tf_evaluation.out','rb')
tf = tf.readlines()
tf_eval = {}
for i in tf:
	i = i.decode('utf-8').strip().split('\t')
	if i[0].strip() in measure:
		if i[1]=='all':
			tf_eval[i[0].strip()] = float(i[2])

tfidf = open('results/tfidf_evaluation.out','rb')
tfidf = tfidf.readlines()
tfidf_eval = {}
for i in tfidf:
	i = i.decode('utf-8').strip().split('\t')
	if i[0].strip() in measure:
		if i[1]=='all':
			tfidf_eval[i[0].strip()] = float(i[2])

rl_fb = open('results/relevance_feedback_evaluation.out','rb')
rl_fb = rl_fb.readlines()
rlfb_eval = {}
for i in rl_fb:
	i = i.decode('utf-8').strip().split('\t')
	if i[0].strip() in measure:
		if i[1]=='all':
			rlfb_eval[i[0].strip()] = float(i[2])



own_m = open('results/own_method_evaluation.out','rb')
own_m = own_m.readlines()
own_m_eval = {}
for i in own_m:
	i = i.decode('utf-8').strip().split('\t')
	if i[0].strip() in measure:
		if i[1]=='all':
			own_m_eval[i[0].strip()] = float(i[2])




benchmarks = ['P_5', 'recall_100','map', 'ndcg']

boolean_out = []
for i in benchmarks:
	boolean_out.append(boolean_eval[i])
tf_out = []
for i in benchmarks:
	tf_out.append(tf_eval[i])
tfidf_out = []
for i in benchmarks:
	tfidf_out.append(tfidf_eval[i])

rlfb_out = []
for i in benchmarks:
	rlfb_out.append(rlfb_eval[i])

own_out = []
for i in benchmarks:
	own_out.append(own_m_eval[i])


xticks = np.arange(len(benchmarks))


fig, ax = plt.subplots(figsize=(10, 7))

ax.bar(xticks, boolean_out, width=0.1, label="Boolean", color="red")
ax.bar(xticks + 0.1, tf_out, width=0.1, label="tf", color="blue")
ax.bar(xticks + 0.2, tfidf_out, width=0.1, label="tf-idf", color="black")
ax.bar(xticks + 0.3, rlfb_out, width=0.1, label="pseudo relevance feedback", color="yellow")
ax.bar(xticks + 0.4, own_out, width=0.1, label="own method", color="green")

ax.set_title("score of different methods under each measurement", fontsize=15)
ax.set_xlabel("Different measures")
ax.set_ylabel("Score ")
ax.legend()

ax.set_xticks(xticks + 0.1)
ax.set_xticklabels(benchmarks)
fig.savefig('results/different_measure.pdf')




P_measure = ['P_5','P_10','P_30','P_100']
R_measure = ['recall_5','recall_10','recall_30','recall_100']

boolean_P_measure = []
for i in P_measure:
	boolean_P_measure.append(boolean_eval[i])
boolean_R_measure = []
for i in R_measure:
	boolean_R_measure.append(boolean_eval[i])

tf_P_measure = []
for i in P_measure:
	tf_P_measure.append(tf_eval[i])
tf_R_measure = []
for i in R_measure:
	tf_R_measure.append(tf_eval[i])

tfidf_P_measure = []
for i in P_measure:
	tfidf_P_measure.append(tfidf_eval[i])
tfidf_R_measure = []
for i in R_measure:
	tfidf_R_measure.append(tfidf_eval[i])

rlfb_P_measure = []
for i in P_measure:
	rlfb_P_measure.append(rlfb_eval[i])
rlfb_R_measure = []
for i in R_measure:
	rlfb_R_measure.append(rlfb_eval[i])

own_P_measure = []
for i in P_measure:
	own_P_measure.append(own_m_eval[i])
own_R_measure = []
for i in R_measure:
	own_R_measure.append(own_m_eval[i])


fig = plt.figure()
ax = plt.axes()
ax.set_title('tradeoff between precision and recall for each method')
plt.setp(ax, xlabel = 'recall')
plt.setp(ax, ylabel = 'precision')

#ax.plot(boolean_P_measure, boolean_R_measure, color="red", linewidth=2.0, linestyle="-",label="boolean")
ax.plot(tf_P_measure, tf_R_measure, color="blue", linewidth=2.0, linestyle="-",label="tf")
ax.plot(tfidf_P_measure, tfidf_R_measure, color="black", linewidth=2.0, linestyle="-",label="tfidf")
ax.plot(rlfb_P_measure, rlfb_R_measure, color="yellow", linewidth=2.0, linestyle="-",label="pseudo relevance feedback")
ax.plot(own_P_measure, own_R_measure, color="green", linewidth=2.0, linestyle="-",label="own method")
ax.legend()
plt.savefig('results/precision_recall.pdf')
















