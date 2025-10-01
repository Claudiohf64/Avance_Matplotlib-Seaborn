import torch
import torch.nn as nn

x = torch.tensor([[[1.],[2.],[3.],[4.],[5.]]])
y = torch.tensor([[[2.],[4.],[6.],[8.],[10.]]])

model=nn.Linear(1,1)

criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

