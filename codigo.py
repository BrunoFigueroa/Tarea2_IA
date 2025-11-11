from sklearn.datasets import fetch_covtype
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle as skshuffle

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import json
import time

# PUNTO 1

# Cargar dataset como DataFrame y sacar muestra aleatoria de 10k filas
data = fetch_covtype(as_frame=True)
df = data.frame
df = df.sample(n=10000, random_state=42)

# Verificar estructura
print(df.shape)
print(df['Cover_Type'].value_counts())

# Pasar de 54 columnas a 10, eliminando los booleanos y dejando solo los continuos.
continuous_cols = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", 
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", 
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
]

# Dividir en X (features, 54 columnas) e y (etiqueta, 1 columna)
#X = df[continuous_cols]
X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']

""" ESTO ERA PARA TEST INICIAL, NO IMPORTA PARA LA TAREA
# Tomar 10 filas aleatorias completas (incluye la etiqueta al final) y exportarlo como JSON
sample_df = df.sample(10, random_state=42)
records = sample_df.to_dict(orient="records")
with open("muestra_10_filas.json", "w") as f:
    json.dump(records, f, indent=4)
print("Archivo 'muestra_10_filas.json' creado con 10 filas del dataset.")
"""

# K means. con proporcion 80-20, con 4 configuraciones diferentes.

print()
print("K-MEANS")

results = []

# Escalar los datos (recomendado para clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar train (80%) y test (20%), sin usar la etiqueta Y en el entrenamiento
X_train, X_test, y_train, y_test = tts(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Definir 4 configuraciones distintas de K-Means
kmeans_configs = [
    {"n_clusters": 3, "init": "random", "n_init": 10, "max_iter": 300},
    {"n_clusters": 5, "init": "random", "n_init": 15, "max_iter": 300},
    {"n_clusters": 7, "init": "random", "n_init": 20, "max_iter": 500},
    {"n_clusters": 9, "init": "random", "n_init": 25, "max_iter": 400},
]

# Entrenar y evaluar cada configuración
for i, cfg in enumerate(kmeans_configs, 1):
    model = KMeans(**cfg, random_state=42)
    model.fit(X_train)
    labels = model.labels_

    sil_score = silhouette_score(X_train, labels)
    results.append({
        "model_type": "KMeans",
        "config": cfg,
        "silhouette": sil_score
    })
    print(f"Config {i}: {cfg}, Silhouette Score = {sil_score:.4f}")


# K means++. con proporcion 80-20, con 4 configuraciones diferentes.

print("\n=== K-MEANS++ (init='k-means++') ===")

kmeanspp_configs = [
    {"n_clusters": 3, "init": "k-means++", "n_init": 10, "max_iter": 300},
    {"n_clusters": 5, "init": "k-means++", "n_init": 20, "max_iter": 300},
    {"n_clusters": 7, "init": "k-means++", "n_init": 15, "max_iter": 500},
    {"n_clusters": 9, "init": "k-means++", "n_init": 25, "max_iter": 400},
]

for i, cfg in enumerate(kmeanspp_configs, 1):
    model = KMeans(**cfg, random_state=42)
    model.fit(X_train)
    labels = model.labels_
    sil_score = silhouette_score(X_train, labels)
    results.append({
        "model_type": "KMeans++",
        "config": cfg,
        "silhouette": sil_score
    })
    print(f"Config {i}: {cfg}, Silhouette Score = {sil_score:.4f}")


# MeanShift. con proporcion 80-20, con 4 configuraciones diferentes.

print("\n=== MEANSHIFT ===")

# MeanShift no usa n_clusters, usa bandwidth, que es la distancia para fusionar puntos.

# Calcular una estimacion base de bandwidth para escalar configuraciones
bandwidth_estimate = estimate_bandwidth(X_train, quantile=0.2, n_samples=500)
print(f"Bandwidth estimado base: {bandwidth_estimate:.4f}")

meanshift_configs = [
    {"bandwidth": bandwidth_estimate * 0.5},
    {"bandwidth": bandwidth_estimate},
    {"bandwidth": bandwidth_estimate * 1.5},
    {"bandwidth": bandwidth_estimate * 2.0},
]

for i, cfg in enumerate(meanshift_configs, 1):
    model = MeanShift(**cfg)
    model.fit(X_train)
    labels = model.labels_

    # Algunos clusters pueden tener un solo punto, manejar ese caso (te odio chatgpt >:c)
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(X_train, labels)
    else:
        sil_score = -1

    results.append({
        "model_type": "MeanShift",
        "config": cfg,
        "silhouette": sil_score
    })
    print(f"Config {i}: {cfg}, Silhouette Score = {sil_score:.4f}, Clusters encontrados = {len(np.unique(labels))}")


# Evaluacion global de las 12 configuraciones
results = sorted(results, key=lambda x: x["silhouette"], reverse=True)
top3 = results[:3]

print("\n=== Top 3 configuraciones globales por Silhouette Score ===")
for res in top3:
    print(res)

# Aplicar al test set
for i, res in enumerate(top3, 1):
    algo = res["model_type"]
    cfg = res["config"]
    print(f"\nModelo {i}: {algo}, Config: {cfg}")

    if algo in ["KMeans", "KMeans++"]:
        model = KMeans(**cfg, random_state=42)
    else:
        model = MeanShift(**cfg)

    model.fit(X_train)
    test_clusters = model.predict(X_test)

    train_clusters = model.labels_
    cluster_labels = {}
    for cluster_id in np.unique(train_clusters):
        mask = train_clusters == cluster_id
        dominant_label = y_train.iloc[mask].mode()[0]
        cluster_labels[cluster_id] = dominant_label

    y_pred = [cluster_labels[c] for c in test_clusters if c in cluster_labels]
    valid_idx = [i for i, c in enumerate(test_clusters) if c in cluster_labels]
    match_ratio = np.mean(np.array(y_pred) == y_test.values[valid_idx])
    print(f"Coincidencia entre etiquetas reales y dominantes = {match_ratio:.4f}")

# PUNTO 2 - Versión final (Jupyter)

# Leer configuraciones desde archivo externo
with open("configs.json", "r") as f:
    configs = json.load(f)["models"]

print(f"Configuraciones cargadas: {len(configs)} modelos\n")

# Subdividir parte del train para evaluación (aún dentro del 80%)
X_train_main, X_train_eval, y_train_main, y_train_eval = tts(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)
classes = np.unique(y_train)

# Crear modelos a partir del archivo
models_meta = []
for cfg in configs:
    loss = "log_loss" if cfg["type"] == "logistic" else "hinge"
    clf = SGDClassifier(
        loss=loss,
        penalty="l2",
        alpha=cfg["alpha"],
        learning_rate=cfg["learning_rate"],
        eta0=cfg["eta0"],
        random_state=42
    )
    models_meta.append({
        "name": cfg["name"],
        "type": cfg["type"],
        "cfg": cfg,
        "clf": clf,
        "epochs_done": 0,
        "max_epochs": cfg["max_epochs"],
        "batch_size": cfg["batch_size"],
        "alive": True,
        "last_eval_acc": None
    })

# Inicializar pesos
for m in models_meta:
    init_batch = min(100, X_train_main.shape[0])
    m["clf"].partial_fit(X_train_main[:init_batch], y_train_main[:init_batch], classes=classes)

# Funcion de entrenamiento por bloque
def train_chunk(model, X_main, y_main, epochs_chunk, batch_size, classes):
    n = X_main.shape[0]
    for ep in range(epochs_chunk):
        X_sh, y_sh = skshuffle(X_main, y_main, random_state=int(time.time() * 1000) % 2**32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb, yb = X_sh[start:end], y_sh[start:end]
            model.partial_fit(Xb, yb, classes=classes)
    y_pred_eval = model.predict(X_train_eval)
    acc = accuracy_score(y_train_eval, y_pred_eval)
    return model, acc

# Entrenamiento iterativo con eliminación del peor modelo cada 5 épocas
epochs_chunk = 5
max_rounds = max(m["max_epochs"] for m in models_meta) // epochs_chunk + 1
print(f"Iniciando entrenamiento paralelo ({len(models_meta)} configuraciones, {epochs_chunk} épocas por ronda)\n")

for round_i in range(max_rounds):
    alive = [m for m in models_meta if m["alive"] and m["epochs_done"] < m["max_epochs"]]
    if len(alive) <= 2:
        print("Menos de 3 modelos activos, deteniendo eliminaciones.")
        break

    print(f"--- Ronda {round_i+1} | modelos activos: {len(alive)} ---")
    futures = {}
    with ThreadPoolExecutor(max_workers=min(len(alive), 4)) as exe:
        for m in alive:
            futures[exe.submit(train_chunk, m["clf"], X_train_main, y_train_main, epochs_chunk, m["batch_size"], classes)] = m
        for fut in as_completed(futures):
            m = futures[fut]
            clf_updated, acc = fut.result()
            m["clf"] = clf_updated
            m["epochs_done"] += epochs_chunk
            m["last_eval_acc"] = acc
            print(f"{m['name']} ({m['type']}) -> acc_train_eval={acc:.4f}")

    # Eliminar el peor modelo
    alive = [m for m in models_meta if m["alive"]]
    if len(alive) <= 2:
        break
    worst = min(alive, key=lambda x: x["last_eval_acc"] or -1.0)
    worst["alive"] = False
    print(f"Eliminado: {worst['name']} ({worst['last_eval_acc']:.4f})\n")

# Seleccionar finalistas
finalists = sorted([m for m in models_meta if m["alive"]], key=lambda x: x["last_eval_acc"] or 0.0, reverse=True)[:2]
print("\nFinalistas:")
for f in finalists:
    print(f"- {f['name']} ({f['type']}) acc_train_eval={f['last_eval_acc']:.4f}")

# Evaluar finalistas en test
results_summary = []
for f in finalists:
    clf = f["clf"]
    y_pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)
    print(f"\n== {f['name']} ({f['type']}) ==")
    print(f"Accuracy test: {acc_test:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    results_summary.append({
        "name": f["name"],
        "type": f["type"],
        "train_eval_acc": f["last_eval_acc"],
        "test_acc": acc_test
    })

results_summary

# Al final nunca use pandas, pero si lo borro, voy a tener que recompilar todo en el jupyter, asi que ahi se queda.