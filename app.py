import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs

# Configuración de la página
st.set_page_config(page_title="K-Means: Pintando los Datos", layout="centered")
st.title("Visualizador de K-Means")
st.write("Mira cómo los datos inician sin color y se adaptan al centroide más cercano.")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración")
n_samples = st.sidebar.slider("Número de puntos de datos", 100, 500, 300, step=50)
k_clusters = st.sidebar.slider("Número de clusters (K)", 2, 6, 3)
seed = st.sidebar.number_input("Semilla aleatoria", value=42)


# --- GENERACIÓN DE DATOS ---
@st.cache_data
def generar_datos(n, k, seed):
    X, _ = make_blobs(n_samples=n, centers=k, cluster_std=1.2, random_state=42)
    return X


X = generar_datos(n_samples, k_clusters, 42)

# --- K-MEANS MANUAL (Micro-pasos) ---
estados = []  # Aquí guardaremos cada 'fotograma' de la animación

# Selección aleatoria de centroides iniciales
np.random.seed(42)
idx_aleatorios = np.random.choice(len(X), k_clusters, replace=False)
centroides = X[idx_aleatorios]

# FOTOGRAMA 1: Inicio absoluto. Datos grises (-1), centroides listos.
estados.append({
    'fase': 'Inicio: Datos sin asignar',
    'centroides': centroides.copy(),
    'etiquetas': np.full(len(X), -1)  # -1 significa "Gris/Sin color"
})

# Simulamos hasta 8 iteraciones para no hacer el slider infinito
for i in range(1, 9):
    # FASE A: Asignación (Los datos "absorben" el color de la estrella más cercana)
    # Calculamos la distancia de cada punto a cada centroide
    distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
    etiquetas = np.argmin(distancias, axis=1)  # Tomamos el índice del más cercano

    estados.append({
        'fase': f'Iteración {i}: Los datos toman color',
        'centroides': centroides.copy(),  # Los centroides aún no se mueven
        'etiquetas': etiquetas.copy()
    })

    # FASE B: Actualización (Las estrellas viajan al centro de su nuevo color)
    nuevos_centroides = []
    for c in range(k_clusters):
        puntos_cluster = X[etiquetas == c]
        if len(puntos_cluster) > 0:
            nuevos_centroides.append(puntos_cluster.mean(axis=0))
        else:
            nuevos_centroides.append(centroides[c])

    nuevos_centroides = np.array(nuevos_centroides)

    estados.append({
        'fase': f'Iteración {i}: Los centroides se mueven',
        'centroides': nuevos_centroides.copy(),
        'etiquetas': etiquetas.copy()  # Los colores de los datos se mantienen igual
    })

    # Si los centroides ya no se mueven, terminamos
    if np.allclose(centroides, nuevos_centroides):
        break
    centroides = nuevos_centroides

# --- INTERFAZ DEL SLIDER ---
max_pasos = len(estados)
paso = st.slider("Control de la animación (Fotograma)", 1, max_pasos, 1)

# Extraer los datos del fotograma actual
estado_actual = estados[paso - 1]
df = pd.DataFrame(X, columns=['X', 'Y'])
df['Cluster'] = estado_actual['etiquetas'].astype(str)
df['Cluster'] = df['Cluster'].replace('-1', 'Sin asignar')

# --- CONFIGURACIÓN DE COLORES ---
color_sequence = px.colors.qualitative.Safe
cluster_color_map = {'Sin asignar': '#B0B0B0'}  # Color gris estético
for c in range(k_clusters):
    cluster_color_map[str(c)] = color_sequence[c % len(color_sequence)]

# --- CREACIÓN DEL GRÁFICO ---
fig = px.scatter(
    df, x='X', y='Y',
    color='Cluster',
    title=f"Fase: {estado_actual['fase']}",
    color_discrete_map=cluster_color_map,
    opacity=0.7,
    labels={'Cluster': 'Grupo'}
)

# Agregar las estrellas
for c in range(k_clusters):
    fig.add_trace(
        go.Scatter(
            x=[estado_actual['centroides'][c, 0]],
            y=[estado_actual['centroides'][c, 1]],
            mode='markers',
            marker=dict(
                symbol='star', size=24,
                color=cluster_color_map[str(c)],
                line=dict(width=2, color='black')
            ),
            name=f'Centroide {c}',
            hoverinfo='text',
            text=[f"Centroide {c}"]
        )
    )

fig.update_layout(
    legend_title_text='Elementos',
    plot_bgcolor='rgba(240,240,240,0.5)',
    xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

