import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Configuración de la página de Streamlit
st.set_page_config(page_title="K-Means: Inicialización Aleatoria", layout="centered")
st.title("Visualizador de K-Means")
st.write("Mira cómo los centroides inician al azar y los datos se pintan inmediatamente según el más cercano.")

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("Configuración")
n_samples = st.sidebar.slider("Número de puntos de datos", 100, 500, 300, step=50)
k_clusters = st.sidebar.slider("Número de clusters (K)", 2, 6, 3)


# --- GENERACIÓN DE DATOS (Caché) ---
@st.cache_data
def generar_datos(n, k,seed):
    X, _ = make_blobs(n_samples=n, centers=k, cluster_std=1.2, random_state=seed)
    return pd.DataFrame(X, columns=['X', 'Y'])


df = generar_datos(n_samples, k_clusters, 42)

# --- EJECUCIÓN DE K-MEANS ITERACIÓN POR ITERACIÓN ---
centroides_historial = []
etiquetas_historial = []

# Forzamos a Scikit-Learn a usar 'random' (puntos aleatorios del dataset como inicio)
# y n_init=1 con la misma semilla para poder trackear el progreso real paso a paso.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=k_clusters, init='random', n_init=1, max_iter=i, random_state=42)
    kmeans.fit(df[['X', 'Y']])

    # En la primera iteración de sklearn con max_iter=1, ya se hizo la primera asignación
    # de etiquetas basada en los centroides aleatorios iniciales.
    if i == 1:
        # Guardamos los centroides iniciales estrictamente aleatorios antes de moverse
        # Para obtener los centros iniciales exactos de la semilla:
        np.random.seed(42)
        idx_aleatorios = np.random.choice(len(df), k_clusters, replace=False)
        centroides_iniciales = df[['X', 'Y']].iloc[idx_aleatorios].values
        centroides_historial.append(centroides_iniciales)
    else:
        # Para el resto de pasos, guardamos los centroides ya calculados de la iteración anterior
        centroides_historial.append(kmeans_antiguo_centroides)

    etiquetas_historial.append(kmeans.labels_)
    kmeans_antiguo_centroides = kmeans.cluster_centers_

# Slider para controlar el paso de la animación
max_pasos = len(centroides_historial)
paso = st.slider("Iteración / Paso del Algoritmo", 1, max_pasos, 1)

# Obtener datos del paso seleccionado
idx = paso - 1
df['Cluster'] = etiquetas_historial[idx].astype(str)
centroides_actuales = centroides_historial[idx]

# --- PALETA DE COLORES CONSISTENTE ---
color_sequence = px.colors.qualitative.Safe
unique_clusters = sorted(df['Cluster'].unique())
cluster_color_map = {cluster: color_sequence[int(cluster) % len(color_sequence)] for cluster in unique_clusters}

# --- CREACIÓN DEL GRÁFICO CON PLOTLY ---
# 1. Graficar los puntos de datos (Coloreados instantáneamente por el centroide más cercano)
fig = px.scatter(
    df, x='X', y='Y',
    color='Cluster',
    title=f"K-Means: Estado en el Paso {paso}",
    color_discrete_map=cluster_color_map,
    opacity=0.6,
    labels={'Cluster': 'Grupo'}
)

# 2. Agregar los centroides individuales con su color correspondiente
for i, cluster in enumerate(unique_clusters):
    # Validar que el índice no desborde el array de centroides actuales
    if i < len(centroides_actuales):
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
                    size=24,  # Estrellas grandes
                    color=color_centroide,  # Mismo color que sus puntos asignados
                    line=dict(width=2, color='black')  # Borde negro para que resalte
                ),
                name=f'Centroide {cluster}',
                hoverinfo='text',
                text=[f"Centroide {cluster}"]
            )
        )

# Ajustes estéticos
fig.update_layout(
    legend_title_text='Elementos',
    plot_bgcolor='rgba(240,240,240,0.5)',
    xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

