import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
def generar_caso_de_uso_comparar_umbrales_precision_recall():
    """
    Genera un caso de uso aleatorio para
    comparar_umbrales_precision_recall(X, y, umbrales).

    Retorna:
        input  (dict): {"X": np.ndarray, "y": np.ndarray, "umbrales": list}
        output (pd.DataFrame): resultado esperado de la función
    """
    rng = np.random.default_rng()

    n_samples = int(rng.integers(120, 300))
    n_features = int(rng.integers(3, 8))

    X = rng.standard_normal(size=(n_samples, n_features))
    coef = rng.standard_normal(size=n_features)
    logit = X @ coef + rng.standard_normal(n_samples) * 0.5
    y = (logit > 0).astype(int)

    n_umbrales = int(rng.integers(4, 9))
    umbrales = sorted(
        np.round(rng.uniform(0.1, 0.9, size=n_umbrales), 2).tolist()
    )

    # --- Calcular output esperado ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]

    filas = []
    for u in umbrales:
        y_pred = (proba >= u).astype(int)
        filas.append({
            "umbral": round(float(u), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        })

    resultado = pd.DataFrame(filas).reset_index(drop=True)

    input_dict = {
        "X": X.copy(),
        "y": y.copy(),
        "umbrales": umbrales,
    }
    return input_dict, resultado
    # --- Ejemplo de uso ---
if __name__ == "__main__":
    
    # 1. Generar un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_comparar_umbrales_precision_recall()
    
    # 2. Mostrar INPUT
    print("=== INPUT ===")
    
    print("\n🔢 Matriz X:")
    print(f"Shape: {entrada['X'].shape}")
    print("Primeras 2 filas:")
    print(entrada["X"][:2])
    
    print("\n🎯 Vector y:")
    print(f"Shape: {entrada['y'].shape}")
    print("Primeros 10 valores:")
    print(entrada["y"][:10])
    
    print("\n⚙️ Umbrales:")
    print(entrada["umbrales"])
    
    # 3. Mostrar OUTPUT esperado
    print("\n=== OUTPUT ESPERADO ===")
    print("Métricas por umbral:")
    print(salida_esperada)
    
    # 4. Información útil
    print("\n=== INFO ===")
    print(f"Número de muestras: {entrada['X'].shape[0]}")
    print(f"Número de features: {entrada['X'].shape[1]}")
    print(f"Número de umbrales evaluados: {len(entrada['umbrales'])}")