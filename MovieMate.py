#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


movies_data=pd.read_csv("tmdb_5000_movies.csv")
credits_data=pd.read_csv("tmdb_5000_credits.csv")
credits_data.head()


# In[3]:


data=movies_data.merge(credits_data,on='title')


# In[4]:


data.head(1)


# In[5]:


data.columns


# In[6]:


#genres'
# movie_id
# keywords
# title
# overview
# cast
# crew


# In[7]:


data=data[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


data.dropna(inplace=True)


# In[12]:


data.isnull().sum()


# In[13]:


data.duplicated().sum()


# In[14]:


import ast


# In[15]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
       
    return L;


# In[16]:


data['genres']=data['genres'].apply(convert)


# In[17]:


data.head(1)


# In[18]:


data['keywords']=data['keywords'].apply(convert)


# In[19]:


data.head(1)


# In[20]:


def convert_cast(obj):
    L=[]
    count=0;
    for i in ast.literal_eval(obj):
        if(count<3):
            L.append(i['name'])
            count+=1
        else: break;
       
    return L;


# In[21]:


data['cast']=data['cast'].apply(convert_cast)


# In[22]:


data.head(1)


# In[23]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break;
    return L;


# In[24]:


data['crew']=data['crew'].apply(fetch_director)


# In[25]:


data.head(3)


# In[26]:


data['overview']=data['overview'].apply(lambda x:x.split())


# In[27]:


data.head(3)


# In[28]:


data['genres']=data['genres'].apply(lambda x:[i.replace(" ","") for i in x])
data['keywords']=data['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
data['cast']=data['cast'].apply(lambda x:[i.replace(" ","") for i in x])
data['crew']=data['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[29]:


data.head()


# In[30]:


data['tags']=data['overview']+data['genres']+data['keywords']+data['cast']+data['crew']


# In[31]:


new_df=data[['movie_id','title','tags']]


# In[32]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[33]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[ ]:





# In[34]:


new_df.head(3)


# In[35]:


# vecterisation 
from sklearn.feature_extraction.text import CountVectorizer


# In[36]:


cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()


# In[37]:


vectors


# In[38]:


cv.get_feature_names_out()[:100]


# In[39]:


import nltk 


# In[40]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[41]:


def stemming(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))  
    return " ".join(y)


# In[42]:


new_df['tags']=new_df['tags'].apply(stemming)


# In[43]:


new_df.head()


# In[44]:


from sklearn.metrics.pairwise import cosine_similarity


# In[45]:


similarity=cosine_similarity(vectors)


# In[46]:


similarity[2]


# In[47]:


sorted(list(enumerate(similarity[0])),key=lambda x:x[1],reverse=True)[1:6]


# In[48]:


def recommendation(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),key=lambda x:x[1],reverse=True)[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    


# In[55]:


recommendation('Inception')

