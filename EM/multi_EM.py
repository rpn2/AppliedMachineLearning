import os
# from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("docword.nips.txt") as nips_file:
  nips_lines = nips_file.read().split("\n")   
with open("vocab.nips.txt") as voc_file:
  vocab = voc_file.read().split("\n")  

num_docs = int(nips_lines[0])
num_words = int(nips_lines[1])
num_topics = 30
nips_lines = np.array([[int(y) for y in x.split(" ")] for x in  nips_lines[3:-1]])   

x_ik = np.zeros((num_docs, num_words))
for line in nips_lines:
  x_ik[line[0]-1][line[1]-1] = line[2]

pi_j = np.random.rand(num_topics)
pi_j = pi_j / np.sum(pi_j)
# kmeans1 = KMeans(n_clusters =num_topics, random_state=0).fit(x_ik)  
# from collections import Counter
# for i in range(num_topics):
#   pi_j[i] = Counter(kmeans1.labels_)[i] / num_docs
 
p_jk = np.random.rand(num_topics, num_words)
# for i, label in enumerate(kmeans1.labels_):
#   p_jk[label, ] += x_ik[i,]                         
p_jk = p_jk / np.sum(p_jk, axis = 1)[:, None]
     

W_ij = np.zeros((num_docs, num_topics))
likeli_list = []  
likeli_list.append(np.zeros(num_topics)) 

for step in range(50):
  print(step)
  print(pi_j)
  for i in range(num_docs):
    for j in range(num_topics):
      W_ij[i, j] = np.sum(x_ik[i,] * np.log(p_jk[j,])) + np.log(pi_j[j])
  W_ij = np.exp(W_ij - W_ij.max(1)[:,None])
  W_ij = W_ij / W_ij.sum(1)[::,None]
  p_jk = np.full((num_topics, num_words), 0.000001 * num_docs)
  for j in range(num_topics):
    for i in range(num_docs):
      p_jk[j, ] += np.dot(x_ik[i,], W_ij[i,j])
  p_jk = p_jk / np.sum(p_jk, axis = 1)[:, None]
  pi_j = np.sum(W_ij, axis = 1) / num_docs

  likelihood = 0
  for i in range(num_docs):
    for j in range(num_topics):
      likelihood += (np.dot(x_ik[i,], np.log(p_jk[j,:])) + np.log(pi_j[j])) * W_ij[i,j]
  likeli_list.append(likelihood)
  #Convergence checking
  if np.linalg.norm(likeli_list[step] - likeli_list[step-1]) < 0.0001:  
    break
     
topic_index = np.zeros(num_docs)
prob = np.zeros(30)
for i in range(num_docs):    
        topic_index[i] = np.argsort(W_ij[i,])[::-1][0]
        for j in range(num_topics):
           if topic_index[i] == j:
                  prob[j] +=  1
prob = prob / num_docs
ax, fig =  plt.subplots()  
plt.plot([i for i in range(1,31)], prob)
plt.xlabel('Topic')
plt.ylabel('Probability') 
plt.title("Probability Chart")
plt.ion()
plt.show()      

freq_word_index = []
for j in range(num_topics):
  freq_word_index.append(np.argsort(p_jk[j,])[::-1][0:10])
freq_word = np.zeros((30,10), dtype = object)
for i in range(num_topics): 
  for j in range(10):
    freq_word[i][j]= vocab[freq_word_index[i][j]]
df = pd.DataFrame(freq_word)
df.to_csv("freq_words.csv")