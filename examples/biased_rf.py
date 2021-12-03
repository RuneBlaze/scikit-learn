from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
from sklearn import tree
X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(random_state=0)
base = regr.base_estimator
base.use_bias = True
base.bias = np.asarray([1,0,0,0], dtype=np.float64)
regr.fit(X, y)
e = regr.estimators_[0]
text_representation = tree.export_text(e)
print(text_representation)
print(regr.predict([[0, 0, 0, 0]]))