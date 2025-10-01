import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = load_iris()
x = iris.data
y = iris.target
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        self.hidden = nn.Linear(4, 16)
        self.output = nn.Linear(16, 3)
    def forward (self,x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

model=IrisNN()
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(),lr=0.01)

for epochs in range(100):
    output=model(x_train)
    loss= criterion(output,y_train)
    optimizer.zero_grad()
    loss.backward
    optimizer.step()
    if epochs% 10==0:
        print(loss.item())

entrada=x_test[0].reshape(1,-1)

with torch.no_grad():
    salida=model(entrada)
    print(salida)
    i,y_pred = torch.max(salida,1)
    print(y_pred)
    print(iris.target_names[y_pred])
