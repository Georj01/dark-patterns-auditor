import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Configuración global
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_COLUMNS = ['categoria_patron', 'sesgo_psicologico']


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    # Advertencias si faltan las columnas reales de tu CSV
    if 'videojuego' not in df.columns:
        logging.warning("Falta la columna 'videojuego'. No habrá gráficos por juego individual.")
    if 'modelo' not in df.columns:
        logging.warning("Falta la columna 'modelo'. No habrá gráficos por F2P o Suscripción.")

    logging.info('Recuento de nulos por columna:\n%s', df.isna().sum().to_string())

    # Normalizar nulos
    df['categoria_patron'] = df['categoria_patron'].fillna('Desconocido').astype(str)
    df['sesgo_psicologico'] = df['sesgo_psicologico'].fillna('Desconocido').astype(str)

    return df


def reporte_basico(df: pd.DataFrame, output_dir: str):
    print('\n--- RESUMEN BÁSICO (GENERAL) ---')
    print(f"Total de registros: {len(df)}")
    print(f"Filas duplicadas: {df.duplicated().sum()}")

    combo = df.groupby(['categoria_patron', 'sesgo_psicologico']).size().sort_values(ascending=False)
    print('\n--- TOP 15: Combinaciones (Patrón + Sesgo) ---')
    print(combo.head(15).to_string())

    # Exportaciones pesadas a CSV
    combo.to_csv(os.path.join(output_dir, 'combo_categoria_sesgo_general.csv'), index=True)
    
    crosstab = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'], margins=True)
    crosstab_norm = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'], normalize='index')

    crosstab.to_csv(os.path.join(output_dir, 'crosstab_absoluto_general.csv'))
    crosstab_norm.to_csv(os.path.join(output_dir, 'crosstab_norm_filas_general.csv'))


def grafico_barras(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, color: str):
    if series.empty:
        return
    plt.figure(figsize=(10, 6))
    series.plot(kind='barh', color=color)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()


def generar_visuales_segmento(df_subset: pd.DataFrame, nombre_segmento: str, output_dir: str):
    if df_subset.empty:
        return
        
    patron_counts = df_subset['categoria_patron'].value_counts().sort_values(ascending=True)
    sesgo_counts = df_subset['sesgo_psicologico'].value_counts().sort_values(ascending=True)

    # Limpiar el nombre para los archivos
    nombre_archivo = nombre_segmento.replace(" ", "_").replace("/", "-")

    grafico_barras(patron_counts, f'Frecuencia de Patrones Oscuros: {nombre_segmento}', 'Número de Impactos', 'Categoría', os.path.join(output_dir, f'patrones_{nombre_archivo}.png'), '#2b8cbe')
    grafico_barras(sesgo_counts, f'Sesgos Psicológicos Explotados: {nombre_segmento}', 'Número de Observaciones', 'Sesgo', os.path.join(output_dir, f'sesgos_{nombre_archivo}.png'), '#e6550d')


def generar_informes_y_graficos(df: pd.DataFrame, output_dir: str):
    print('\n--- GENERANDO GRÁFICOS PNG ---')
    
    # 1. Dataset General
    generar_visuales_segmento(df, 'General', output_dir)
    print("- PNGs de la muestra General creados.")

    # 2. Por Modelo de Monetización (F2P vs Suscripcion)
    if 'modelo' in df.columns:
        modelos = df['modelo'].dropna().unique()
        for modelo in modelos:
            df_modelo = df[df['modelo'] == modelo]
            generar_visuales_segmento(df_modelo, f'Modelo_{modelo}', output_dir)
        print(f"- PNGs por Modelo de Monetización ({', '.join(modelos)}) creados.")

    # 3. Por Juego Individual
    if 'videojuego' in df.columns:
        juegos = df['videojuego'].dropna().unique()
        for juego in juegos:
            df_juego = df[df['videojuego'] == juego]
            generar_visuales_segmento(df_juego, f'Juego_{juego}', output_dir)
        print(f"- PNGs por Juego individual ({', '.join(juegos)}) creados.")

    # Mapa de calor global
    numeric_cols = df.select_dtypes(include='number')
    if numeric_cols.shape[1] > 1:
        corr = numeric_cols.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Mapa de Correlaciones General')
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap_general.png'), dpi=600, bbox_inches='tight')
        plt.close()


def apoyo_confianza_lift(df: pd.DataFrame, output_dir: str):
    total = len(df)
    if total == 0:
        return

    marg_p = df['categoria_patron'].value_counts()
    marg_s = df['sesgo_psicologico'].value_counts()
    combo = df.groupby(['categoria_patron', 'sesgo_psicologico']).size().reset_index(name='n')

    combo['soporte'] = combo['n'] / total
    combo['conf_patron_a_sesgo'] = combo['n'] / combo['categoria_patron'].map(marg_p)
    combo['conf_sesgo_a_patron'] = combo['n'] / combo['sesgo_psicologico'].map(marg_s)
    
    prob_p = combo['categoria_patron'].map(marg_p) / total
    prob_s = combo['sesgo_psicologico'].map(marg_s) / total
    combo['lift'] = combo['soporte'] / (prob_p * prob_s)

    combo = combo.sort_values(by=['lift', 'conf_patron_a_sesgo', 'soporte'], ascending=False)
    combo.to_csv(os.path.join(output_dir, 'reglas_asociacion_completas.csv'), index=False)

    print('\n--- TOP 20: REGLAS DE ASOCIACIÓN GENERAL (Ordenadas por Lift) ---')
    print(combo.head(20).to_string(index=False, formatters={
        'soporte': '{:.3f}'.format,
        'conf_patron_a_sesgo': '{:.3f}'.format,
        'conf_sesgo_a_patron': '{:.3f}'.format,
        'lift': '{:.3f}'.format,
    }))


def estadistico_chi_cuadrado(df: pd.DataFrame, output_dir: str):
    cross = pd.crosstab(df['categoria_patron'], df['sesgo_psicologico'])
    
    if cross.empty or cross.shape[0] < 2 or cross.shape[1] < 2:
        logging.warning("Matriz insuficiente para calcular Chi-cuadrado.")
        return

    chi2, p, dof, expected = chi2_contingency(cross)
    total = cross.values.sum()
    min_dim = min(cross.shape[0] - 1, cross.shape[1] - 1)
    cramer_v = np.sqrt(chi2 / (total * min_dim)) if total > 0 and min_dim > 0 else np.nan

    print('\n=== CONCLUSIÓN ESTADÍSTICA (CHI-CUADRADO GENERAL) ===')
    print(f'Chi2: {chi2:.2f} | P-value: {p:.6g} | Grados de libertad: {dof} | Cramer V: {cramer_v:.3f}')

    if p < 0.05:
        print('-> Resultado: SIGNIFICATIVO. Hay relación real entre patrón y sesgo en el cómputo global.')
    else:
        print('-> Resultado: NO SIGNIFICATIVO. Parecen independientes a nivel global.')

    expected_df = pd.DataFrame(expected, index=cross.index, columns=cross.columns)
    row_mass = cross.sum(axis=1) / total
    col_mass = cross.sum(axis=0) / total
    denominator = np.sqrt(expected_df * (1 - row_mass.values.reshape(-1, 1)) * (1 - col_mass.values.reshape(1, -1)))
    adj_residuals = (cross - expected_df) / denominator.replace(0, np.nan)

    expected_df.to_csv(os.path.join(output_dir, 'chi2_matriz_esperada.csv'))
    adj_residuals.to_csv(os.path.join(output_dir, 'chi2_residuales_ajustados.csv'))


def main():
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_input_defecto = os.path.join(directorio_script, 'Datos.csv')
    ruta_output_defecto = os.path.join(directorio_script, 'output')

    parser = argparse.ArgumentParser(description='Auditoría de Patrones Oscuros. Segmentación automática de imágenes.')
    parser.add_argument('--input', default=ruta_input_defecto, help='Ruta al CSV (separador ;).')
    parser.add_argument('--output', default=ruta_output_defecto, help='Carpeta para exportar CSVs y PNGs.')
    args = parser.parse_args()

    if not os.path.isabs(args.input):
        args.input = os.path.join(directorio_script, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(directorio_script, args.output)

    if not os.path.exists(args.input):
        logging.error('El archivo "%s" no existe. Abortando.', args.input)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    logging.info('Procesando datos...')

    try:
        df = pd.read_csv(args.input, sep=';')
        df = validate_dataframe(df)
    except Exception as e:
        logging.error('Fallo crítico al preparar los datos: %s', e)
        sys.exit(1)

    reporte_basico(df, args.output)
    generar_informes_y_graficos(df, args.output)
    apoyo_confianza_lift(df, args.output)
    estadistico_chi_cuadrado(df, args.output)

    print(f'\n[+] Ejecución completada. Tienes tus PNGs segmentados y los CSVs duros en: {args.output}')


if __name__ == '__main__':
    main()