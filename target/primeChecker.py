import math
import sys
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


def generar_batches(inicio, fin, tamaño_batch):
    """Genera tuplas (inicio, fin) para dividir el rango."""
    return [(i, min(i + tamaño_batch, fin + 1)) for i in range(inicio, fin + 1, tamaño_batch)]


def main():
    # --- Leer parámetros desde línea de comandos ---
    if len(sys.argv) < 2:
        print("Uso: python code0_seq.py <num_digitos>")
        print("Ejemplo: python code0_seq.py 4")
        sys.exit(1)

    num_digitos = int(sys.argv[1])

    # --- Parámetros del problema ---
    tamaño_batch = 10
    inicio = 10 ** (num_digitos - 1)
    fin = (10 ** num_digitos) - 1

    # --- Generar los batches ---
    batches = generar_batches(inicio, fin, tamaño_batch)
    print(f"\nContando primos de {num_digitos} dígitos (rango {inicio}-{fin})")
    print(f"Total de batches: {len(batches)}\n")

    # --- Ejecución secuencial ---
    t0 = time.perf_counter()
    resultados = [contar_primos_en_batch(b) for b in batches]
    total_primos = sum(resultados)
    elapsed = time.perf_counter() - t0

    # --- Resultados ---
    print(f"\n✅ Total de primos con {num_digitos} dígitos: {total_primos}")
    print(f"⏱️ Tiempo total: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
