# IPython log file


from gala import classify
datas = []
labels = []
import numpy as np
list(map(np.shape, labels))
for i in range(3, 4):
    data, label = classify.load_training_data_from_disk('training-data-%i.h5' % i, names=['data', 'labels'])
    datas.append(data)
    labels.append(label[:, 0])
    
X0 = np.concatenate(datas, axis=0)
y0 = np.concatenate(labels)
# runtime was 5min for 3000 samples, expect ~2h for 72,000
# for 280,000, expect ~8h (took 10h)
idx = np.random.choice(len(y0), size=280000, replace=False)
X, y = X0[idx], y0[idx]
param_dist = {'n_estimators': [20, 100, 200, 500],
              'max_depth': [3, 5, 20, None],
              'max_features': ['auto', 5, 10, 20],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}
from sklearn import grid_search as gs
from time import time
from sklearn import ensemble
ensemble.RandomForestClassifier().get_params().keys()
rf = ensemble.RandomForestClassifier()
random_search = gs.GridSearchCV(rf, param_grid=param_dist, refit=False,
                                verbose=2, n_jobs=12)
start=time(); random_search.fit(X, y); stop=time()
