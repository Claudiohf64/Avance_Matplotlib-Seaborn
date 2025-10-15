import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
import numpy as np
import torch
import torch.nn as nn

datos = pd.DataFrame(
    {
        "leng": ["Java", "PHP", "Java", "PHP", "Java"],  # NOMINAL
        "cert": ["NO", "NO", "SI", "SI", "NO"],  # NOMINAL
        "rang": ["J", "J", "S", "S", "S"],  # ORDINAL
        "contratado": [0, 0, 1, 0, 0],  # TARGET
    }
)

encoder = LabelEncoder()
x_num = encoder.fit_transform(datos[["rang"]]).reshape(-1, 1)

one_encoder = OneHotEncoder(sparse_output=False)
x_cat = one_encoder.fit_transform(datos[["cert", "leng"]])

label_bin = LabelBinarizer()
y_num = label_bin.fit_transform(datos["contratado"])

x = np.hstack([x_num, x_cat])
y = np.array(y_num)

ToTorch_x = torch.tensor(x, dtype=torch.float32)
ToTorch_y = torch.tensor(y, dtype=torch.float32)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(5, 3)
        self.output = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


model = SimpleNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epochs in range(1000):
    output = model(ToTorch_x)
    loss = criterion(output, ToTorch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    y_pred = model(ToTorch_x)
    print(y_pred.round())

print(ToTorch_y)


