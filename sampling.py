
import pandas as pd
import numpy as np
dataseries = pd.Series(np.random.rand(30))
dataseries.head()

datawithfeatures=pd.DataFrame(np.random.randn(50,4),columns=list("TARS"))
datawithfeatures.head()
sampleseries = dataseries.sample(n=4)
sampleseries1=dataseries.sample(frac=0.3,replace=True)
