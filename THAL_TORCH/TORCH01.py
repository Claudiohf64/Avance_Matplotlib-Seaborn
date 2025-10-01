# * **Torch:** Biblioteca Deep Learning
# * **nn:**  Modulo de Torch que representa una RNA
import torch
import torch.nn as nn

# * Torch estructura la data de la RNA en arreglos multidimensionales con la capacidad del calculo de derivadas
# * Por defecto el tipo de dato es Float----------------------
x = torch.tensor([[1.],[2.],[3.],[4.],[5.]])
y= torch.tensor([[2.],[4.],[6.],[8.],[10.]])

# ** Estructura de RNA **
# * Por lo general una REG. LINEAL no requiere una estructura compleja, nuestra RNA tiene una capa oculta que a su vez es la capa de salida, dicha capa contiene 1 neurona
model= nn.Linear(1,1)

# * **criterion:** Nuestra funcion de perdida (MSE), que trabaja mejor en problemas de REG. LINEAL
# * **optimizer:** SGD para optimizar el trabajo de la funcion de perdida, trabaja mejor en problemas de REG. LINEAL
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

# * 200 EPOCAS
# * **outputs:** Calculamos la salida de nuestro modelo (entrenamiento)
# * **loss:** Obtenemos la perdida (Diferencia entre las salidas del modelo y las etiquetas reales)
# * **zero_grad():** Limpia las gradientes de la epoca anterior
# * **backward:** Retropropagacion
# * **step():** Actualiza los pesos y sesgos
for epoch in range (200):
    outputs=model(x)
    loss=criterion(outputs,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch% 10 == 0:
        print(loss.item())

# * **no_grad()**: Indicamos a torch que deje de calcular gradientes
# * Obtenemos en **y_pred** una prediccion con un dato nuevo
with torch.no_grad():
    y_pred = model(torch.tensor([[15.]]))
    print(y_pred)
print(x,y,model)
