import pandas as pd
import matplotlib.pyplot as plt

escuela = pd.DataFrame({"id_escuela": [1, 2], "nombre_escuela": ["ETI", "ENI"]})

carrera = pd.DataFrame(
    {
        "id_carrera": [1, 2, 3, 4],
        "nombre_carrera": ["Ing. Software", "Redes", "Administracion", "Contabilidad"],
        "id_escuela": [1, 1, 2, 2]
    }
)

print("---Escuela---")
print(escuela)
print("---Carreras---")
print(carrera)
reporte1 = pd.merge(carrera, escuela, on="id_escuela", how="inner")
# SELECT c.nombre,e.nombre FROM carrera c INNER JOIN escuela e ON e.id=c.id_escuela GROUP BY e.nombre
print(reporte1)
conteo=reporte1.groupby('nombre_escuela')['id_escuela'].count()
print(conteo)
plt.bar(conteo.index, conteo.values)
plt.show()