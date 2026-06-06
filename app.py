import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Configuración de la página de Streamlit
st.set_page_config(page_title="K-Means Interactivo por Colores", layout="centered")
st.title("Visualizador de K-Means Paso a Paso 🤖🌈")
st.write("Observa cómo cada centroide (estrella) viaja desde el centro hacia su grupo de datos del mismo color.")

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("Configuración")
n_samples = st.sidebar.slider("Número de puntos de datos", 100, 500, 300, step=50)
k_clusters = st.sidebar.slider("Número de clusters (K)", 2, 6, 3)
seed = st.sidebar.number_input("Semilla aleatoria (para cambiar los datos)", value=42)


# --- GENERACIÓN DE DATOS (Caché) ---
@st.cache_data
def generar_datos(n, k, seed):
    # Genera grupos de datos realistas (blobs)
    X, _ = make_blobs(n_samples=n, centers=k, cluster_std=1.2, random_state=seed)
    return pd.DataFrame(X, columns=['X', 'Y'])


df = generar_datos(n_samples, k_clusters, seed)

# --- EJECUCIÓN DE K-MEANS CUSTOMIZADA (Centroides con color) ---
centroides_historial = []
etiquetas_historial = []

# Calcular el centro exacto de TODOS los datos para el inicio
centro_x = df['X'].mean()
centro_y = df['Y'].mean()

# Paso 1: Forzamos el inicio en el centro con ruido aleatorio
np.random.seed(seed)
centroides_iniciales = np.array([
    [centro_x + np.random.uniform(-0.1, 0.1), centro_y + np.random.uniform(-0.1, 0.1)]
    for _ in range(k_clusters)
])
centroides_historial.append(centroides_iniciales)

# Etiquetas Paso 1
kmeans_paso1 = KMeans(n_clusters=k_clusters, init=centroides_iniciales, n_init=1, max_iter=1)
kmeans_paso1.fit(df[['X', 'Y']])
etiquetas_historial.append(kmeans_paso1.labels_)

# Pasos del 2 al 10: Evolución
for i in range(1, 10):
    kmeans = KMeans(n_clusters=k_clusters, init=centroides_iniciales, n_init=1, max_iter=i)
    kmeans.fit(df[['X', 'Y']])
    centroides_historial.append(kmeans.cluster_centers_)
    etiquetas_historial.append(kmeans.labels_)

# Slider para controlar el paso de la animación
max_pasos = len(centroides_historial)
paso = st.slider("Iteración / Paso del Algoritmo", 1, max_pasos, 1)

# Obtener datos del paso seleccionado
idx = paso - 1
df['Cluster'] = etiquetas_historial[idx].astype(str)  # Etiquetas como texto para colores discretos
centroides_actuales = centroides_historial[idx]

# --- DEFINICIÓN DE PALETA DE COLORES FIJA ---
# Usamos una paleta discreta para asegurarnos de que el cluster "0" siempre sea del mismo color, etc.
# Plotly Express usa por defecto colores de 'Safe' o 'D3'.
color_sequence = px.colors.qualitative.Safe  # Esta paleta es amigable para daltónicos

# Mapeo manual para asegurar consistencia entre puntos y centroides
# Creamos una lista ordenada de clusters únicos para asignar colores
unique_clusters = sorted(df['Cluster'].unique())
# Creamos un diccionario: {'0': '#color1', '1': '#color2', ...}
cluster_color_map = {cluster: color_sequence[i % len(color_sequence)] for i, cluster in enumerate(unique_clusters)}

# --- CREACIÓN DEL GRÁFICO CON PLOTLY ---
# 1. Graficar los puntos de datos coloreados por su cluster actual
fig = px.scatter(
    df, x='X', y='Y',
    color='Cluster',
    title=f"K-Means: Estado en la iteración {paso}",
    color_discrete_map=cluster_color_map,  # Usamos el mapeo manual
    opacity=0.6,
    labels={'Cluster': 'Grupo Data'}
)

# 2. Agregar los centroides COMO TRAZAS INDIVIDUALES (Para asignar colores específicos)
for i, cluster in enumerate(unique_clusters):
    # Obtener el centroide actual para este cluster específico
    # (Scikit-learn ordena los centroides por el ID del cluster, del 0 al K-1)
    centroide_x = centroides_actuales[i, 0]
    centroide_y = centroides_actuales[i, 1]

    color_centroide = cluster_color_map[cluster]

    fig.add_trace(
        go.Scatter(
            x=[centroide_x],
            y=[centroide_y],
            mode='markers',
            marker=dict(
                symbol='star',
                size=22,  # Más grande para que destaque
                color=color_centroide,  # Color del grupo
                line=dict(width=2, color='black')  # Borde negro para contraste
            ),
            name=f'Centroide {cluster}',  # Nombre en la leyenda
            hoverinfo='text',
            text=[f"Centroide {cluster}"]
        )
    )

# Ajustes estéticos del gráfico
fig.update_layout(
    legend_title_text='Elementos',
    plot_bgcolor='rgba(240,240,240,0.5)',
    xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
    hovermode='closest'
)

# Mostrar el gráfico interactivo en Streamlit
st.plotly_chart(fig, use_container_width=True)

# Mensaje dinámico explicativo
if paso == 1:
    st.warning(
        f"📍 **Paso 1:** Los {k_clusters} centroides (estrellas de colores) han comenzado agrupados en el centro geométrico. Ya tienen asignado el color del grupo que van a rastrear.")
else:
    st.success(
        f"🚀 **Paso {paso}:** ¡Cada centroide viaja hacia su núcleo! Observa cómo la estrella {cluster_color_map[unique_clusters[0]]} (color del Grupo {unique_clusters[0]}) se acerca al centro exacto de la nube de puntos del mismo color.")