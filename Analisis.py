import pandas as pd
import matplotlib.pyplot as plt
# Cargar los datos desde el archivo CSV
data = pd.read_csv('datos.csv')
# Mostrar las primeras filas del DataFrame para verificar la carga de datos
print(data.head())
# Análisis de datos: Estadísticas descriptivas
print(data.describe())