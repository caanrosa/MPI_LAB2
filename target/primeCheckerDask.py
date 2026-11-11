import math
import sys
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import time


def es_primo(n: int) -> bool:
    """Verifica si un número es primo."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def contar_primos_en_batch(batch):
    """Cuenta cuántos primos hay en un lote de números."""
    inicio, fin = batch
    cuenta = sum(1 for n in range(inicio, fin) if es_primo(n))
    #print(f"Procesado batch {inicio}-{fin-1} → {cuenta} primos")
    return cuenta


@delayed
def delayed_contar_primos_en_batch(batch):
    """Versión delayed (para ejecución paralela)."""
    return contar_primos_en_batch(batch)


def generar_batches(inicio, fin, tamaño_batch):
    """Genera tuplas (inicio, fin) para dividir el rango."""
    return [(i, min(i + tamaño_batch, fin + 1)) for i in range(inicio, fin + 1, tamaño_batch)]


def main():
    # --- Leer parámetros desde línea de comandos ---
    if len(sys.argv) < 3:
        print("Uso: python code0.py <num_digitos> <n_workers>")
        print("Ejemplo: python code0.py 4 3")
        sys.exit(1)

    num_digitos = int(sys.argv[1])
    n_workers = int(sys.argv[2])

    # --- Configuración del cluster local ---
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1,processes=True )
    client = Client(cluster)
    print(f"\nCluster Dask iniciado con {n_workers} workers")
    print(f"Dashboard: {client.dashboard_link}")

    # --- Parámetros del problema ---
    tamaño_batch = 10
    inicio = 10 ** (num_digitos - 1)
    fin = (10 ** num_digitos) - 1

    # --- Generar los batches ---
    batches = generar_batches(inicio, fin, tamaño_batch)
    print(f"\nContando primos de {num_digitos} dígitos (rango {inicio}-{fin})")
    print(f"Total de batches: {len(batches)}\n")

    # --- Crear tareas retrasadas (delayed) ---
    tareas = [delayed_contar_primos_en_batch(b) for b in batches]

    # --- Ejecutar en paralelo ---
    t0 = time.perf_counter()
    resultados = compute(*tareas)
    total_primos = sum(resultados)
    elapsed = time.perf_counter() - t0

    # --- Resultados ---
    print(f"\n✅ Total de primos con {num_digitos} dígitos: {total_primos}")
    print(f"⏱️ Tiempo total: {elapsed:.3f} s")

    client.close()


if __name__ == "__main__":
    main()
