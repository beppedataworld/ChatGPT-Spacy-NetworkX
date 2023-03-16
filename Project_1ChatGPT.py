import os
import requests
import json
import pandas as pd
from pandas import json_normalize
import datetime as dt 

url = 'https://content.guardianapis.com/search'
MY_API = os.environ.get("API_GUARDIAN")
my_params= {"api-key":MY_API, "q":"chatgpt",  "format":"json", "page":1, "page-size":100}

# input("topic")
response=requests.get(url,params=my_params).json()

res_dict=json_normalize(response['response']['results'])
df = pd.DataFrame(res_dict).reset_index(drop=True)

chat_gpt = df.convert_dtypes() #in realtà potrei non doverlo fare. Infatti se uso 'loc' in ogni colonna mi dirà reale type
chat_gpt['webPublicationDate']=pd.to_datetime(chat_gpt['webPublicationDate'],format="%Y-%m-%d")
chat_gpt['webPublicationDate']=chat_gpt['webPublicationDate'].dt.date
#ATTENTO: il fatto che chat_gpt['webPublicationDate'].dtype ti dia 'object' non significa che non sia 'datetime'.
#Infatti se fai: chat_gpt.loc[10,'webPublicationDate'] ti darà 'datetime'. Confermato su stackoverflow
chat_gpt.sort_values('webPublicationDate',ascending=False,inplace=True)
#NUMBER OF ARTICLES AT DAY
n_art=chat_gpt.groupby('webPublicationDate')['id'].nunique().reset_index(name='Articles for days')
n_art.set_index('webPublicationDate')['Articles for days'].tail(30).plot(kind='bar',figsize=(15,5))

'''-----------------PART 2: Tokenization and Wordclouds-------------------'''
import spacy
from spacy import displacy

nlp=spacy.load("en_core_web_sm")
ruler=nlp.add_pipe("entity_ruler",before='ner')
patterns1=[{'label':'Skynet','pattern':'ChatGPT'}]
patterns2=[{'label':'Skynet', 'pattern':[{'TEXT':{'REGEX':r"\b(AI)\b"}}]}]
patterns3=[{'label':'Skynet','pattern':[{'TEXT':{'REGEX':r"\b(chatbot)\b"}}]}]
patterns4=[{'label':'Skynet', 'pattern':[{'TEXT':{'REGEX':r"\b(Bard)\b"}}]}]
ruler.add_patterns(patterns1)
ruler.add_patterns(patterns2)
ruler.add_patterns(patterns3)
ruler.add_patterns(patterns4)
entity_list=[]
chat_gpt_doc=list(nlp.pipe(chat_gpt['webTitle']))
for doc in chat_gpt_doc:
    for s in doc.sents:
        entities=[e.text for e in s.ents]
        labels=[e.label_ for e in s.ents]
        entity_list.append({'Sents':s,'Entities':entities,'Labels':labels})
df_entity=pd.DataFrame(entity_list)
df_entity = df_entity[df_entity['Entities'].map(len)>0]
df_entity.head(10)


for e in chat_gpt_doc:
    displacy.render(e,style='ent')

'''WORDCLOUDS'''
import numpy as np
import os
from PIL import Image
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt


list_words=[]
for e in df_entity['Entities']:
    list_words.extend(e)

lab=[]
for l in df_entity['Labels']:
    lab.extend(l)

mystop=["John Naughton","Australia","Turkey","Syria","Guardian"]

for i,v in enumerate(lab):
    for j,k in enumerate(list_words):
        if k in mystop:
            list_words.pop(j)
            lab.pop(i)

d=os.getcwd()
stop=set(STOPWORDS)
stop.update(mystop) #stop.add(str(mystop))
my_mask=np.array(Image.open(d+"/chatgpt-logo.png"))
wc=WordCloud(stopwords=stop,mask=my_mask,background_color='black',contour_width=2,contour_color='white').generate(str(list_words))
wc.to_image().show()
#Alternative:
# import matplotlib.pyplot as plt
# plt.imshow(wc,interpolation='bilinear',
# cmap=plt.cm.cividis)
# plt.axis('off')
# plt.figure()
# plt.show()


# ENTITIES: COUNT & COMBINATIONs
from collections import Counter,OrderedDict
a=dict(Counter(list_words).items())
b=OrderedDict(sorted(a.items(),key=lambda x:x[1],reverse=True))
series = pd.Series(b)


import itertools 
from itertools import combinations
combo=[]
for e in df_entity['Entities']:
    l=list(combinations(e,2))
    combo.extend(l)

c=dict(Counter(combo).items())

duplicates = set()
for key_tuple in list(c.keys()):
        if key_tuple in duplicates or (key_tuple[1], key_tuple[0]) in duplicates:
            c.pop(key_tuple)
        duplicates.add(key_tuple)
o=OrderedDict(sorted(c.items(),key=lambda x:x[1],reverse=True))
series_combo = pd.Series(o)

plt.rcParams['font.size'] = 25
fig,ax=plt.subplots(1,2,figsize=(18,10))
fig.set_facecolor('steelblue')
ax[0].set_facecolor('palegoldenrod')
ax[1].set_facecolor('palegoldenrod')

s1=series[:5].plot(kind='bar',color='gray',ax=ax[0])
s2=series_combo[:5].plot(kind='bar',color='gray',ax=ax[1])

ax[0].bar_label(s1.containers[0])
ax[1].bar_label(s2.containers[0])

plt.suptitle('Counts & Combos',color='purple',size='x-large',style='oblique')
ax[0].set_title('Top Tags',color='firebrick')
ax[1].set_title('Top Tag-Pairs',color='firebrick')

ax[0].tick_params(labelrotation=25)
ax[1].tick_params(labelrotation=25)
plt.show()
'''--------------NETWORKX----------------------'''
import networkx as nx

#Creare df_combo(Source-Target)
df_combo=pd.DataFrame.from_dict(series_combo).reset_index()
df_combo.rename(columns={'level_0':'NodeA','level_1':'NodeB',0:'Values'},inplace=True)

G=nx.from_pandas_edgelist(df_combo,source='NodeA',target='NodeB',edge_attr='Values')

'''Network Visualization (PYVIS)'''
from pyvis.network import Network
n_degree=dict(G.degree())
nx.set_node_attributes(G,n_degree,'size')
plt.rcParams["figure.figsize"]=(15,10)
# mypos=nx.kamada_kawai_layout(G)
# nx.draw_networkx(G,arrows=True,pos=mypos)
net = Network(notebook = False, width="1000px", height="700px", bgcolor='#222222', font_color='white')
net.from_nx(G)
net.show("Skynet.html")
#We see ChatGPT shows the most of connections.
#We can prove it by checking its 'degrees':
sorted(G.degree,key=lambda x: x[1],reverse=True)

'''----CHECKS-----'''
# G.nodes
# G.edges
# G['ChatGPT']['AI']['Values']
# G.nodes['Bard']['size']

'''-------------Community-----------------'''
'''Method N.1'''
import community.community_louvain as community_louvain
groups = community_louvain.best_partition(G) 

nx.set_node_attributes(G, groups, 'group')
group_net = Network(notebook = False, width="1000px", height="700px", bgcolor='#222222', font_color='white')
group_net.from_nx(G)
group_net.show("skynet_communities.html")

'''Method N.2'''
centrality = nx.betweenness_centrality(G)

lpc = nx.community.label_propagation_communities(G)
community_index = {n: i for i, com in enumerate(lpc) for n in com}

#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(G)
node_color = [community_index[n] for n in G]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    G,
    pos=pos,
    with_labels=True,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

#If we want to check how different are groups
#created by these 2 methods:
sorted(centrality.items(),key=lambda x: x[1],reverse=True)
sorted(groups.items(),key=lambda x: x[1],reverse=True)

'''Graph Section-'''
entity_list2=[]
for i2,doc in enumerate(chat_gpt_doc):
    for s in doc.sents:
        entities=[e.text for e in s.ents]
        labels=[e.label_ for e in s.ents]
        section=chat_gpt['sectionName'][i2]
        entity_list2.append({'Sents':s,'Entities':entities,'Labels':labels,'Section':section})
df_entity2=pd.DataFrame(entity_list2)
df_entity2 = df_entity2[df_entity2['Entities'].map(len)>0]
df_entity.head(10)

ent = [x2 for x in df_entity2['Entities'] for x2 in x]
sec = [x for x in df_entity2['Section']]

nod = list(set(ent+sec))
ed=[]
a=zip(df_entity2['Entities'],df_entity2['Section'])
for i in a:
    for i2 in i[0]:
        ed.append((i2,i[1]))

plt.figure(figsize=(20,20))
g = nx.Graph()


g.add_nodes_from(ent, label='entities')
g.add_nodes_from(sec, label='section')
g.add_edges_from(ed)

labels = {}
for node in nod:
    if node in sec:
        labels[node] = node
#alternativa
labels={x:x for x in nod}

pos = nx.spring_layout(g)

nx.draw_networkx_nodes(g,pos=pos,
                       nodelist=ent,
                       node_color='darkblue',
                       node_size=10,
                       alpha=0.5)

nx.draw_networkx_nodes(g,pos=pos,
                       nodelist=sec,
                       node_color='lightblue',
                       node_size=50,
                       alpha=0.5)

nx.draw_networkx_edges(g,pos=pos,
                       edgelist=ed,
                       width=1,alpha=0.5,edge_color='grey')

nx.draw_networkx_labels(g, pos=pos,
                        labels=labels)
plt.show()