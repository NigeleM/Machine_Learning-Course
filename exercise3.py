
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
datasetsample = load_boston()
X = datasetsample.data
Y = datasetsample.target

print(X)
print(Y)

regression = LinearRegression()
genmodel = regression.fit(X,Y)
testcoef = genmodel.coef_
print(genmodel.coef_)
