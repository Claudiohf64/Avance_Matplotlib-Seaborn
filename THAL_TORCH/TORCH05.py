import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

datos = pd.DataFrame(
    {
        'exp': [1, 2, 3, 4, 5],
        'rango': ['j', 'j', 'j', 's', 's'],
        'contratado': [1, 1, 1, 0, 0],
    }
)

x=datos[['exp','rango']]

scaler=StandardScaler()
x_num=scaler.fit_transform(x[['exp']])

encoder=LabelEncoder()
x_cat=encoder.fit_transform(x['rango']).reshape(-1,1)

x=np.hstack([x_num,x_cat])
y=np.array(datos[['contratado']])

ToTorch_x= torch.tensor(x,dtype=torch.float32)
ToTorch_y= torch.tensor(y,dtype=torch.float32)

class SimpleNN(nn.Module):
    def __init__ (self):
        super(SimpleNN,self).__init__()
        self.output=nn.Linear(2,1)
        self.sigmoid=nn.Sigmoid()
    def forward (self,ToTorch_x):
        x=self.sigmoid(self.output(ToTorch_x))
        return x
model=SimpleNN()
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for epochs in range(200):
    output=model(ToTorch_x)
    loss=criterion(output,ToTorch_y)
    loss.backward
    optimizer.step
with torch.no_grad():
    y_pred=model(torch.tensor([[3.,0.]]))
    print(y_pred)



    