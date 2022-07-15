
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
X = iris.data
print(X.shape)

X_std = StandardScaler().fit_transform(X)
covmat = np.cov(X_std.T)
print(covmat)

eigval,eigvec = np.linalg.eig(covmat)
print(eigvec)
print(eigval)

eigpairs = [(np.abs(eigval[i]),eigvec[:,i]) for i in range(len(eigval))]
total = sum(eigval)
varexp = [(i/total)*100 for i in sorted(eigval,reverse=True)]
print(varexp)
