import pandas as pd
import matplotlib.pyplot as plt
# Cargar los datos desde el archivo CSV
data = pd.read_csv('datos.csv')
# Mostrar las primeras filas del DataFrame para verificar la carga de datos
print(data.head())
# Análisis de datos: Estadísticas descriptivas
print(data.describe())
# Análisis de datos: Distribución de una variable (por ejemplo, 'Edad')
plt.hist(data['Edad'], bins=10, edgecolor='black')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()
# Análisis de datos: Relación entre dos variables (por ejemplo, 'Edad' y 'Ingresos')
plt.scatter(data['Edad'], data['Ingresos'])
plt.title('Relación entre Edad e Ingresos')
plt.xlabel('Edad')
plt.ylabel('Ingresos')
plt.show()
# Análisis de datos: Correlación entre variables