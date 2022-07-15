
from sklearn import linear_model
X = [[1.,2.],[2.,2.],[3.,3.],[4.,4.]]
Y = [1.,2.,3.,4.]

reg = linear_model.BayesianRidge()
reg.fit(X,Y)
tempval = reg.predict([[1,0.]])
