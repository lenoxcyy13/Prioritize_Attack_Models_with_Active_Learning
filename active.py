import numpy as np
import pandas as pd
import json
from sklearn import preprocessing
import matplotlib.pylab as plt

f = open('json file')
file = json.load(f)
num = []
for i in file:
    num.append(i)
data = []
for i in range(len(num)):
    for j in file[num[i]]['data']:
        data.append(j)     
f.close()
df = pd.DataFrame(data)

df.columns = ['x(Attack-Stage)', 'y(Port-Service)', 'bubble_size', 'effective_evidence_count', 'model_id', 'x_feature_entropy', 'y_feature_entropy', 'x_seriousness', 'y_open_frequency']
df = df[df['x(Attack-Stage)'] != 'null']
df = df[df['y(Port-Service)'] != 'null']
df = df[df['y_open_frequency'] != 0]
df = df.drop(columns=['bubble_size', 'model_id'])
df.reset_index(drop=True, inplace=True)
print(df)
all = np.arange(len(df))

df_s = df.groupby('y(Port-Service)')['y_open_frequency'].mean()
l_s = list(pd.DataFrame(df_s)['y_open_frequency'])
s = sum(l_s)/len(l_s)

#x_seriousness*y_open_frequency
a = df['x_seriousness']
b = list(df['y_open_frequency'])
b = np.log10(b)
y_raw = []
for i in range(len(a)):
    if ((a[i]*0.8)+(b[i]*0.2)) > ((17*0.8)+(s*0.2)):
        y_raw.append(1)
    else:
        y_raw.append(0)


X_raw = df.iloc[:,2:6].to_numpy()
X_raw = preprocessing.scale(X_raw)
y_raw = np.array(y_raw)

n_labeled_examples = X_raw.shape[0]
training_indices = np.random.choice(n_labeled_examples, size=50, replace=False)
X_train = X_raw[training_indices]
y_train = y_raw[training_indices]
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)


from sklearn import svm
from modAL.models import ActiveLearner
model = svm.SVC(C=10, probability=True)
learner = ActiveLearner(estimator=model, X_training=X_train, y_training=y_train)
prediction = learner.predict(X_raw)
is_correct = (prediction == y_raw)
unqueried_score = learner.score(X_raw, y_raw)

q_index = []
N_QUERIES = 15
performance_history = [unqueried_score]
for index in range(N_QUERIES):
     query_index, query_instance = learner.query(X_pool)
     X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
     learner.teach(X=X, y=y)
     X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
     model_accuracy = learner.score(X_raw, y_raw)
     print('Accuracy after query {n}: {acc:0.4f}'.format(n=index+1, acc=model_accuracy))
     performance_history.append(model_accuracy)
     q_index.append(int(query_index))

result = []
for i in range(len(X_pool)):
     data_for_prediction_array = (X_pool[i].reshape(1,-1))
     result.append(learner.predict_proba(data_for_prediction_array))
con = []
con_1 = []
for i in range(len(result)):
     if (result[i][0][0] > result[i][0][1]):
          con.append(result[i][0][0])
          con_1.append(result[i][0][1])
     else:
          con.append(result[i][0][1])
          con_1.append(result[i][0][1])
all = [i for i in list(all) if i not in list(training_indices)]
for i in q_index:
     del all[i]
data = {
     'Attack-Stage' : df.iloc[all,0],
     'Port-Service' : df.iloc[all,1],
     'critical' : y_pool,
     'confidence' : con,
     'confidence_of_1' : con_1
}
df_r = pd.DataFrame(data, columns = ['Attack-Stage', 'Port-Service', 'critical', 'confidence', 'confidence_of_1'])
print()
print(df_r.iloc[:,0:4])

from sklearn.inspection import permutation_importance
r = permutation_importance(learner, X_pool, y_pool, n_repeats=10, random_state=0)
for i in r.importances_mean.argsort()[::-1]:
          print(f"{df.iloc[:,2:7].columns.tolist()[i]:<8}"
                f'    '
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")

df_A = df_r.groupby(by = 'Attack-Stage')['confidence_of_1'].mean()
df_P = df_r.groupby(by = 'Port-Service')['confidence_of_1'].mean()
print()
print(df_A.sort_values())
print()
print(df_P.sort_values())
print()
print(pd.DataFrame(df_r.groupby(['Attack-Stage', 'Port-Service']).size()))
