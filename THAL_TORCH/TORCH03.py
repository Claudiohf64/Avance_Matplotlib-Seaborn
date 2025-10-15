import torch
import torch.nn as nn

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)
        self.sigmoid= nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.hidden(x))
        x =self.sigmoid(self.output(x))
        return x

model = SimpleNN()
criterion=nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range (1000):
    outputs=model(x)
    loss=criterion(outputs,y)
    optimizer.zero_grad()
    loss.backward
    optimizer.step()
    if epoch%100==0:
        print(loss.item())
        print(outputs.round())

with torch.no_grad():
    y_pred = model(x)
    print(y_pred.round())

    