import torch
import torch.nn as nn
import numpy as np
x = np.linspace(-10,10,200).reshape(-1,1)
y = np.array([[1 if i < 0 else 0 for i in x]] ).reshape(-1,1)
ToTensor_x = torch.tensor(x,dtype=torch.float32)
ToTensor_y= torch.tensor(y,dtype=torch.float32).reshape(-1,1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.output = nn.Linear(1, 1)
        self.sigmoid= nn.Sigmoid()
    def forward(self,ToTensor_x):
        x =self.sigmoid(self.output(ToTensor_x))
        return x

model = SimpleNN()
criterion=nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range (1000):
    outputs=model(ToTensor_x)
    loss=criterion(outputs,ToTensor_y)
    optimizer.zero_grad()
    loss.backward
    optimizer.step()
    # if epoch%100==0:
    #     print(loss.item())
    #     print(outputs.round())

with torch.no_grad():
    y_pred = model(torch.tensor([[-20.],[-5.],[2.],[8.],[2.],[10.],[0.],]))
    print(y_pred.round())

