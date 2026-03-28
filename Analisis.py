import argparse
import logging
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración global
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_COLUMNS = ['categoria_patron', 'sesgo_psicologico']


def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    logging.info('Recuento de valores nulos por columna:')
    logging.info('\n%s', df.isna().sum())

    # Normalizar y reemplazar nulos para no romper conteos
    df['categoria_patron'] = df['categoria_patron'].fillna('Desconocido').astype(str)
    df['sesgo_psicologico'] = df['sesgo_psicologico'].fillna('Desconocido').astype(str)

    return df


def reporte_basico(df: pd.DataFrame, output_dir: str):
    print('\nResumen de dataframe:')
    print(df.describe(include='all').transpose())

    dupes = df.duplicated().sum()
    print(f'Número de filas duplicadas: {dupes}')

    combo = df.groupby(['categoria_patron', 'sesgo_psicologico']).size().sort_values(ascending=False)
    print('\nConteo por combinación categoria_patron + sesgo_psicologico:')
    print(combo.head(15).to_string())

    combo.to_csv(os.path.join(output_dir, 'combo_categoria_sesgo.csv'), index=True)


def grafico_barras(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, color: str):
    plt.figure(figsize=(10, 6))
    series.plot(kind='barh', color=color)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    logging.info('Gráfico guardado en %s', filename)


def generar_informes(df: pd.DataFrame, output_dir: str):
    patron_counts = df['categoria_patron'].value_counts().sort_values(ascending=True)
    sesgo_counts = df['sesgo_psicologico'].value_counts().sort_values(ascending=True)

    print('\nFrecuencia de Patrones Oscuros:')
    print(patron_counts.to_string())
    print('\nSesgos Psicológicos Explotados:')
    print(sesgo_counts.to_string())

    patron_counts.to_csv(os.path.join(output_dir, 'frecuencia_patrones.csv'))
    sesgo_counts.to_csv(os.path.join(output_dir, 'frecuencia_sesgos.csv'))

    grafico_barras(
        patron_counts,
        'Frecuencia de Patrones Oscuros en Fortnite',
        'Número de Impactos',
        'Categoría del Patrón Oscuro',
        os.path.join(output_dir, 'grafico_patrones.png'),
        '#2b8cbe',
    )

    grafico_barras(
        sesgo_counts,
        'Sesgos Psicológicos Explotados en Fortnite',
        'Número de Observaciones',
        'Sesgo Cognitivo',
        os.path.join(output_dir, 'grafico_sesgos.png'),
        '#e6550d',
    )

    if df.select_dtypes(include='number').shape[1] > 0:
        corr = df.select_dtypes(include='number').corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.title('Mapa de Calor de Correlaciones (variables numéricas)')
        plt.savefig(heatmap_path, dpi=600, bbox_inches='tight')
        plt.close()
        logging.info('Heatmap de correlación guardado en %s', heatmap_path)
    else:
        logging.info('No hay columnas numéricas para generar mapa de calor de correlación.')


def main():
    parser = argparse.ArgumentParser(description='Análisis de patrones oscuros en Fortnite con gráficas y reportes.')
    parser.add_argument('--input', default='Datos.csv', help='Ruta al archivo CSV de entrada (separador ;).')
    parser.add_argument('--output', default='output', help='Directorio de salida para gráficos y CSV resultantes.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logging.info('Cargando datos desde %s', args.input)
    df = pd.read_csv(args.input, sep=';')

    df = validate_dataframe(df)

    reporte_basico(df, args.output)
    generar_informes(df, args.output)

    print('\n¡Análisis completado! Gráficos y reportes exportados con éxito en %s' % args.output)


if __name__ == '__main__':
    main()