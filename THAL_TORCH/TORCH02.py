# 1) Ajustar el código del Ejercicio1, para que la celda Nro. 2, trabaje con NUMPY.
# 2) Implementar una RNA torch con 2 entradas y una salida, debe calcular la suma de entradas 

import torch
import torch.nn as nn
import numpy as np


x = np.array([[1.,2.],[3.,4.],[5.,6],[7.,8.],[9.,10.]])
y= np.array([x[:,0]+x[:,1]])

ToTensor_x=torch.tensor(x,dtype=torch.float)
ToTensor_y=torch.tensor(y,dtype=torch.float).reshape(-1,1)

model= nn.Linear(2,1)

criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range (1000):
    outputs=model(ToTensor_x)
    loss=criterion(outputs,ToTensor_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch% 10 == 0:
        print('Perdida: \n',loss.item() ,'\n')

with torch.no_grad():
    y_pred = model(torch.tensor([[11.,12.]]))
    print('Prediccion: \n', y_pred ,'\n')

print('tensor X \n',ToTensor_x,'\n')
print('tensor Y \n',ToTensor_y)
