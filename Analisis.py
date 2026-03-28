import argparse
import logging
import os
from datetime import datetime

import numpy as np
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

    # Cruces adicionales con porcentajes
    crosstab = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'], margins=True, normalize=False)
    crosstab_norm = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'], normalize='index')
    crosstab_norm_col = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'], normalize='columns')

    crosstab.to_csv(os.path.join(output_dir, 'crosstab_categoria_sesgo.csv'))
    crosstab_norm.to_csv(os.path.join(output_dir, 'crosstab_categoria_sesgo_norm_row.csv'))
    crosstab_norm_col.to_csv(os.path.join(output_dir, 'crosstab_categoria_sesgo_norm_col.csv'))

    print('\nCrosstab base (con totales):')
    print(crosstab)
    print('\nCrosstab normalizado por fila (suma 1 en cada categoría de patrón):')
    print(crosstab_norm)
    print('\nCrosstab normalizado por columna (suma 1 en cada sesgo):')
    print(crosstab_norm_col)


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


def apoyo_confianza_lift(df: pd.DataFrame, output_dir: str):
    # Calcula soporte, confianza y lift para cruces entre categorias y sesgos
    total = len(df)
    marg_p = df['categoria_patron'].value_counts()
    marg_s = df['sesgo_psicologico'].value_counts()
    combo = df.groupby(['categoria_patron', 'sesgo_psicologico']).size().reset_index(name='n')

    combo['soporte'] = combo['n'] / total
    combo['confianza_patron_a_sesgo'] = combo['n'] / combo['categoria_patron'].map(marg_p)
    combo['confianza_sesgo_a_patron'] = combo['n'] / combo['sesgo_psicologico'].map(marg_s)
    combo['lift'] = combo['soporte'] / ((combo['categoria_patron'].map(marg_p) / total) * (combo['sesgo_psicologico'].map(marg_s) / total))

    combo = combo.sort_values(by=['lift', 'confianza_patron_a_sesgo', 'soporte'], ascending=False)
    combo.to_csv(os.path.join(output_dir, 'asociacion_patron_sesgo.csv'), index=False)

    print('\nReglas de asociación (top 20 por lift):')
    print(combo.head(20).to_string(index=False, formatters={
        'soporte': '{:.3f}'.format,
        'confianza_patron_a_sesgo': '{:.3f}'.format,
        'confianza_sesgo_a_patron': '{:.3f}'.format,
        'lift': '{:.3f}'.format,
    }))


def estadistico_chi_cuadrado(df: pd.DataFrame, output_dir: str):
    from scipy.stats import chi2_contingency

    cross = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'])
    chi2, p, dof, expected = chi2_contingency(cross)

    total = cross.values.sum()
    n_rows, n_cols = cross.shape
    cramer_v = np.sqrt(chi2 / (total * min(n_rows - 1, n_cols - 1))) if total > 0 else np.nan

    row_mass = cross.sum(axis=1) / total
    col_mass = cross.sum(axis=0) / total
    expected_df = pd.DataFrame(expected, index=cross.index, columns=cross.columns)

    adj_residuals = (cross - expected_df) / np.sqrt(
        expected_df * (1 - row_mass.values.reshape(-1, 1)) * (1 - col_mass.values.reshape(1, -1))
    )

    result = {
        'chi2': chi2,
        'p_value': p,
        'degrees_of_freedom': dof,
        'cramer_v': cramer_v,
        'observed_sum': total,
    }

    with open(os.path.join(output_dir, 'chi2_test.txt'), 'w', encoding='utf-8') as f:
        f.write('Chi-cuadrado para categoria_patron vs sesgo_psicologico\n')
        f.write(f'chi2: {chi2:.6f}\n')
        f.write(f'p-value: {p:.6g}\n')
        f.write(f'degrees_of_freedom: {dof}\n')
        f.write(f'Cramer V: {cramer_v:.6f}\n')
        f.write('Observaciones totales: %d\n\n' % total)

        f.write('=== Matriz observada ===\n')
        f.write(cross.to_string())
        f.write('\n\n=== Matriz esperada ===\n')
        f.write(expected_df.to_string())
        f.write('\n\n=== Residuales ajustados ===\n')
        f.write(adj_residuals.round(3).to_string())

    logging.info('Prueba de chi-cuadrado guardada en %s', os.path.join(output_dir, 'chi2_test.txt'))

    excel_path = os.path.join(output_dir, 'analisis_estadistico.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        cross.to_excel(writer, sheet_name='observado')
        expected_df.to_excel(writer, sheet_name='esperado')
        (cross - expected_df).to_excel(writer, sheet_name='residuals')
        adj_residuals.to_excel(writer, sheet_name='adj_residuals')

    logging.info('Reporte de chi-cuadrado en Excel guardado en %s', excel_path)

    print('\nResultados de chi-cuadrado:')
    print(f'chi2 = {chi2:.6f}, p-value = {p:.6g}, dof = {dof}, cramer_v = {cramer_v:.6f}')

    if p < 0.05:
        print('Resultado significativo: se rechaza la hipótesis nula de independencia (p < 0.05).')
    else:
        print('No significativo: no se puede rechazar la hipótesis nula de independencia (p >= 0.05).')

    return result


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
    apoyo_confianza_lift(df, args.output)
    estadistico_chi_cuadrado(df, args.output)

    # Cruces automáticos entre categorías adicionales (si existen)
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in REQUIRED_COLUMNS]
    if cat_cols:
        logging.info('Se encontraron columnas categóricas adicionales para cruces: %s', cat_cols)
        for col in cat_cols:
            cross = pd.crosstab(df[col], df['categoria_patron'], normalize='index')
            cross.to_csv(os.path.join(args.output, f'crosstab_{col}_a_categoria_patron_norm.csv'))
            print(f'Cruce normalizado {col} -> categoria_patron guardado: crosstab_{col}_a_categoria_patron_norm.csv')
    else:
        logging.info('No hay columnas categóricas adicionales para cruces.')

    print('\n¡Análisis completado! Gráficos y reportes exportados con éxito en %s' % args.output)


if __name__ == '__main__':
    main()