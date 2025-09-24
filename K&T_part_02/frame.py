import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
frame = pd.DataFrame(x,y)
frame['type']=cancer.target_names[cancer.target]
print(frame)
