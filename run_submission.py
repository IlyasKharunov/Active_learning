#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gzip
import xgboost as xgb
from time import time


# In[ ]:


def expand_data(test_data, selected_features = None, verbose = True, To_Torch = False):
    if selected_features is None:
        selected_features = np.arange(11,236)
    else:
        selected_features = selected_features + 1
    test_data_e = np.empty((len(test_data),len(selected_features) + 11), dtype = test_data.dtype)
    test_data_e[:,:11] = test_data
    features_add = []
    features_sub = []
    features_multiply = []
    features_divide1 = []
    features_divide2 = []
    k = 11
    for i in range(1, 11):
        for j in range(i+1,11):
            if k in selected_features:
                features_add.append([i,j,k])
            if k+1 in selected_features:
                features_sub.append([i,j,k+1]) 
            if k+2 in selected_features:
                features_multiply.append([i,j,k+2])
            if k+3 in selected_features:
                features_divide1.append([i,j,k+3])
            if k+4 in selected_features:
                features_divide2.append([j,i,k+4])
            k += 5
    k = 11
    if len(selected_features) == 225:
        for i in range(45):
            test_data_e[:,k] = test_data_e[:,features_add[i][0]] + test_data_e[:,features_add[i][1]]
            test_data_e[:,k+1] = test_data_e[:,features_sub[i][0]] - test_data_e[:,features_sub[i][1]]
            test_data_e[:,k+2] = test_data_e[:,features_multiply[i][0]] * test_data_e[:,features_multiply[i][1]]
            test_data_e[:,k+3] = (test_data_e[:,features_divide1[i][0]] + 1e-7) / (test_data_e[:,features_divide1[i][1]] + 1e-7)
            test_data_e[:,k+4] = (test_data_e[:,features_divide2[i][0]] + 1e-7) / (test_data_e[:,features_divide2[i][1]] + 1e-7)
            k += 5
        if verbose:
            for i in range(45):
                print(f'{features_add[i][2]-1} feature is adding x{features_add[i][0]} and x{features_add[i][1]}')
                print(f'{features_sub[i][2]-1} feature is substracting x{features_sub[i][1]} from x{features_sub[i][0]}')
                print(f'{features_multiply[i][2]-1} feature is multiplying x{features_multiply[i][0]} and x{features_multiply[i][1]}')
                print(f'{features_divide1[i][2]-1} feature is dividing x{features_divide1[i][0]} by x{features_divide1[i][1]}')
                print(f'{features_divide2[i][2]-1} feature is dividing x{features_divide2[i][0]} by x{features_divide2[i][1]}')

    else:
        for pair in features_add:
            test_data_e[:,k] = test_data_e[:,pair[0]] + test_data_e[:,pair[1]]
            k += 1
            if verbose:
                print(f'{pair[2]-1} feature is adding x{pair[0]} and x{pair[1]}')
        for pair in features_sub:
            test_data_e[:,k] = test_data_e[:,pair[0]] + test_data_e[:,pair[1]]
            k += 1
            if verbose:
                print(f'{pair[2]-1} feature is substracting x{pair[1]} from x{pair[0]}')
        for pair in features_multiply:
            test_data_e[:,k] = test_data_e[:,pair[0]]*test_data_e[:,pair[1]]
            k += 1
            if verbose:
                print(f'{pair[2]-1} feature is multiplying x{pair[0]} and x{pair[1]}')
        for pair in features_divide1:
            test_data_e[:,k] = (test_data_e[:,pair[0]] + 1e-7)/(test_data_e[:,pair[1]] + 1e-7)
            k += 1
            if verbose:
                print(f'{pair[2]-1} feature is dividing x{pair[0]} by x{pair[1]}')
        for pair in features_divide2:
            test_data_e[:,k] = (test_data_e[:,pair[0]] + 1e-7)/(test_data_e[:,pair[1]] + 1e-7)
            k += 1
            if verbose:
                print(f'{pair[2]-1} feature is dividing x{pair[0]} by x{pair[1]}')

    if To_Torch:
        te_x = torch.from_numpy(test_data_e[:,1:]).to('cuda').float()
        te_y = torch.from_numpy(test_data_e[:,0]).to('cuda').view((-1,1)).float()
        return te_x, te_y
    else:
        return test_data_e


# In[ ]:


with gzip.open('points.gz','r') as f:
    text = f.read().decode("utf-8")
text = text.split('\n')
text = [line.split() for line in text]
text.pop()
for i in range(len(text)):
    for j in range(len(text[i])-1):
        text[i][j] = float(text[i][j][2:])
    text[i][-1] = float(text[i][-1][3:])
data_to_answer = np.array(text)
data_to_answer = np.concatenate((np.zeros(len(data_to_answer)).reshape(-1,1), 
                                 data_to_answer), axis = 1)


# In[ ]:


ultimate_features = np.array([93,92,75])


# In[ ]:


param = {'objective': 'reg:squarederror'}
param['nthread'] = 4
param['verbosity'] = 0


# In[ ]:


data_to_answer_e = expand_data(data_to_answer, selected_features = ultimate_features)


# In[ ]:


danswer = xgb.DMatrix(data_to_answer_e[:,1:])


# In[ ]:


bst = xgb.Booster(param)  # init model
bst.load_model('test1m.model')


# In[ ]:


#start = time()
answers = bst.predict(danswer)
#print('time required to answer: ', time() - start)


# In[ ]:


with open('final_submission.txt','w') as fout:
    fout.write('Id,Expected\n')
    for i in range(len(answers)):
        fout.write(f'{i+1},{answers[i]}\n')

