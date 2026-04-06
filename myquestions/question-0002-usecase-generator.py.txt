import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
def generar_caso_de_uso_enriquecer_catalogo_productos():
    """
    Genera un caso de uso aleatorio para
    enriquecer_catalogo_productos(df_ventas, df_catalogo).

    Retorna:
        input  (dict): {"df_ventas": ..., "df_catalogo": ...}
        output (pd.DataFrame): resultado esperado de la función
    """
    rng = np.random.default_rng()

    n_productos = int(rng.integers(8, 20))
    all_ids = [f"P{100 + i}" for i in range(n_productos)]
    nombres = [f"Producto_{pid}" for pid in all_ids]
    precios = np.round(rng.uniform(5.0, 500.0, size=n_productos), 2)
    stocks = rng.integers(0, 200, size=n_productos)

    df_catalogo = pd.DataFrame({
        "producto_id": all_ids,
        "nombre": nombres,
        "precio": precios,
        "stock_actual": stocks,
    })

    anio_reciente = 2024
    anio_viejo = 2023
    n_ventas = int(rng.integers(20, 60))

    ids_ventas = [str(x) for x in rng.choice(all_ids + ["P999", "P998"], size=n_ventas)]
    unidades = rng.integers(1, 50, size=n_ventas)

    fechas = []
    for _ in range(n_ventas):
        anio = int(rng.choice([anio_reciente, anio_viejo]))
        mes = int(rng.integers(1, 13))
        dia = int(rng.integers(1, 29))
        fechas.append(f"{anio}-{mes:02d}-{dia:02d}")

    idx_nulos = rng.choice(n_ventas, size=int(rng.integers(1, 4)), replace=False)
    fechas_arr = np.array(fechas, dtype=object)
    fechas_arr[idx_nulos] = None

    df_ventas = pd.DataFrame({
        "producto_id": ids_ventas,
        "fecha": fechas_arr,
        "unidades_vendidas": unidades,
    })

    # --- Calcular output esperado ---
    dv = df_ventas.copy()
    dv["fecha"] = pd.to_datetime(dv["fecha"], errors="coerce")
    dv = dv.dropna(subset=["fecha"])

    anio_max = dv["fecha"].dt.year.max()
    dv = dv[dv["fecha"].dt.year == anio_max]

    merged = dv.merge(df_catalogo, on="producto_id", how="inner")
    merged["ingreso_fila"] = merged["unidades_vendidas"] * merged["precio"]

    agg = (
        merged.groupby("producto_id", as_index=False)
        .agg(
            nombre=("nombre", "first"),
            ingresos_anio=("ingreso_fila", "sum"),
            total_unidades_vendidas=("unidades_vendidas", "sum"),
            stock_actual=("stock_actual", "first"),
        )
    )

    agg["rotacion"] = np.where(
        agg["stock_actual"] == 0,
        np.inf,
        agg["total_unidades_vendidas"] / agg["stock_actual"],
    )

    resultado = (
        agg[["producto_id", "nombre", "ingresos_anio",
             "total_unidades_vendidas", "stock_actual", "rotacion"]]
        .sort_values("ingresos_anio", ascending=False)
        .reset_index(drop=True)
    )

    input_dict = {"df_ventas": df_ventas.copy(), "df_catalogo": df_catalogo.copy()}
    return input_dict, resultado
    # --- Ejemplo de uso ---
if __name__ == "__main__":
    
    # 1. Generar un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_enriquecer_catalogo_productos()
    
    # 2. Mostrar INPUT
    print("=== INPUT ===")
    
    print("\n📦 df_catalogo (primeras 5 filas):")
    print(entrada["df_catalogo"].head())
    
    print("\n🧾 df_ventas (primeras 5 filas):")
    print(entrada["df_ventas"].head())
    
    # 3. Mostrar OUTPUT esperado
    print("\n=== OUTPUT ESPERADO ===")
    print("DataFrame enriquecido:")
    print(salida_esperada.head())
    
    # 4. Información útil
    print("\n=== INFO ===")
    print(f"Número de productos en catálogo: {len(entrada['df_catalogo'])}")
    print(f"Número de ventas: {len(entrada['df_ventas'])}")
    print(f"Número de productos en resultado final: {len(salida_esperada)}")
    print("Columnas del resultado:", list(salida_esperada.columns))