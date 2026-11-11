from mpi4py import MPI
import math
import sys
import time

# ------------------------------
# Funciones auxiliares
# ------------------------------
def es_primo(n: int) -> bool:
    """Determina si un número es primo."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limite = int(math.sqrt(n)) + 1
    for i in range(3, limite, 2):
        if n % i == 0:
            return False
    return True


def contar_primos_en_rango(inicio: int, fin: int) -> int:
    """Cuenta los números primos en el rango [inicio, fin)."""
    cuenta = 0
    for n in range(inicio, fin):
        if es_primo(n):
            cuenta += 1
    return cuenta


# ------------------------------
# Programa principal MPI
# ------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Uso: mpiexec -n <N> python code0_mpi.py <num_digitos>")
        sys.exit(0)

    num_digitos = int(sys.argv[1])
    batch_size = 10  # tamaño de cada lote

    # Rango de números según número de dígitos
    inicio_total = 10 ** (num_digitos - 1)
    fin_total = 10 ** num_digitos

    # Dividir en lotes de tamaño batch_size
    batches = [(i, min(i + batch_size, fin_total)) for i in range(inicio_total, fin_total, batch_size)]

    # Distribuir lotes entre procesos
    lotes_locales = [b for i, b in enumerate(batches) if i % size == rank]

    # Sincronización de tiempo
    comm.Barrier()
    t0 = MPI.Wtime()

    # Calcular localmente
    cuenta_local = 0
    for inicio, fin in lotes_locales:
        cuenta_local += contar_primos_en_rango(inicio, fin)

    # Reducir resultados
    total_primos = comm.reduce(cuenta_local, op=MPI.SUM, root=0)

    t1 = MPI.Wtime()

    if rank == 0:
        print(f"\n=== RESULTADO MPI ===")
        print(f"Número de dígitos: {num_digitos}")
        print(f"Total de primos encontrados: {total_primos}")
        print(f"Tiempo total: {t1 - t0:.3f} segundos")
        print(f"Procesos usados: {size}")


if __name__ == "__main__":
    main()
