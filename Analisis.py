import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configurar el rigor estético de los gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# 2. Cargar tu base de datos
# Asegúrate de usar sep=';' porque exportaste desde Excel en español
df = pd.read_csv('Daots_Corregido.csv', sep=';')

# 3. GENERAR GRÁFICO 1: Frecuencia de Patrones Oscuros
plt.figure(figsize=(10, 6))
# Contamos y ordenamos de menor a mayor para que el gráfico de barras horizontales quede perfecto
patron_counts = df['categoria_patron'].value_counts().sort_values(ascending=True)

# Dibujamos las barras (en azul sobrio)
patron_counts.plot(kind='barh', color='#2b8cbe')
plt.title('Frecuencia de Patrones Oscuros en Fortnite', fontsize=14, weight='bold')
plt.xlabel('Número de Impactos', fontsize=12)
plt.ylabel('Categoría del Patrón Oscuro', fontsize=12)

# Exportamos en alta resolución (300 dpi) para que no se pixele en el PDF del TFG
plt.savefig('grafico_patrones.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. GENERAR GRÁFICO 2: Sesgos Psicológicos Explotados
plt.figure(figsize=(10, 6))
sesgo_counts = df['sesgo_psicologico'].value_counts().sort_values(ascending=True)

# Dibujamos las barras (en naranja para contrastar)
sesgo_counts.plot(kind='barh', color='#e6550d')
plt.title('Sesgos Psicológicos Explotados en Fortnite', fontsize=14, weight='bold')
plt.xlabel('Número de Observaciones', fontsize=12)
plt.ylabel('Sesgo Cognitivo', fontsize=12)

# Exportamos la imagen
plt.savefig('grafico_sesgos.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Análisis completado! Gráficos exportados con éxito.")