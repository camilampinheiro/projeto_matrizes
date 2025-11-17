import socket
import pickle
import numpy as np
import threading
import time

# ==========================
# Configuração
# ==========================

# Endereços dos servidores (2 servidores)
SERVERS = [
    ("127.0.0.1", 5001),
    ("127.0.0.1", 5002),
]

# Dimensão das matrizes
# A será de tamanho (N x N) e B de (N x N)
N = 200  # Você pode aumentar (ex.: 500) para testar mais carga

# ==========================
# Funções auxiliares de rede
# ==========================
def recv_all(sock, n_bytes):
    data = b""
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            raise ConnectionError("Conexão interrompida durante o recebimento.")
        data += packet
    return data

def recv_pickle(sock):
    header = recv_all(sock, 8)
    msg_len = int.from_bytes(header, byteorder="big")
    data = recv_all(sock, msg_len)
    return pickle.loads(data)

def send_pickle(sock, obj):
    data = pickle.dumps(obj)
    header = len(data).to_bytes(8, byteorder="big")
    sock.sendall(header + data)

# ==========================
# Versão Serial (não distribuída)
# ==========================

def serial_matrix_mult(A, B):
    """
    Multiplicação serial (tradicional) usando NumPy.
    Aqui não há distribuição nem paralelismo de rede.
    """
    return np.dot(A, B)

# ==========================
# Versão Distribuída (Cliente + 2 Servidores)
# ==========================

def worker(server_index, subA, B, results_dict):
    """
    Thread worker: conecta ao servidor, envia subA e B,
    recebe o resultado parcial e guarda em results_dict[server_index].
    """
    host, port = SERVERS[server_index]
    print(f"[CLIENTE] Conectando ao Servidor {server_index + 1} em {host}:{port}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        print(f"[CLIENTE] Enviando submatriz A{server_index + 1} (shape {subA.shape}) e matriz B (shape {B.shape})...")
        send_pickle(s, (subA, B))

        result = recv_pickle(s)
        print(f"[CLIENTE] Resultado parcial recebido do Servidor {server_index + 1} com shape {result.shape}.\n")

        results_dict[server_index] = result

def distributed_matrix_mult(A, B):
    """
    Multiplicação de matrizes distribuída entre 2 servidores.
    A é dividida em 2 partes de linhas (A1 e A2), cada servidor
    calcula a sua parte: C1 = A1 * B, C2 = A2 * B.
    O cliente depois concatena C1 e C2.
    """
    # Divide A em 2 partes (linhas)
    submatrices_A = np.array_split(A, len(SERVERS), axis=0)

    results = {}
    threads = []

    # Cria uma thread para cada servidor
    for i in range(len(SERVERS)):
        t = threading.Thread(target=worker, args=(i, submatrices_A[i], B, results))
        threads.append(t)

    print("[CLIENTE] Iniciando threads para comunicação com os servidores...\n")

    # Inicia todas as threads
    for t in threads:
        t.start()

    # Espera todas terminarem
    for t in threads:
        t.join()

    print("[CLIENTE] Todos os resultados parciais foram recebidos. Concatenando...\n")

    # Ordena os resultados pelo índice do servidor e empilha verticalmente
    partial_results = [results[i] for i in sorted(results.keys())]
    C = np.vstack(partial_results)

    return C

# ==========================
# Função principal
# ==========================
def main():
    # Gera matrizes A e B aleatórias (inteiros entre 0 e 9)
    np.random.seed(42)  # para reprodutibilidade
    A = np.random.randint(0, 10, size=(N, N))
    B = np.random.randint(0, 10, size=(N, N))

    print("=======================================")
    print("              Resultados              ")
    print("=======================================\n")

    print(f"Dimensões das matrizes:")
    print(f"A: {A.shape}")
    print(f"B: {B.shape}\n")

    # -----------------------------
    # Execução Serial
    # -----------------------------
    print(">>> EXECUÇÃO SERIAL (sem distribuição)")
    start_serial = time.perf_counter()
    C_serial = serial_matrix_mult(A, B)
    end_serial = time.perf_counter()
    tempo_serial = end_serial - start_serial
    print(f"Tempo serial: {tempo_serial:.6f} segundos\n")

    # -----------------------------
    # Execução Distribuída
    # -----------------------------
    print(">>> EXECUÇÃO DISTRIBUÍDA (cliente + 2 servidores)")
    start_dist = time.perf_counter()
    C_dist = distributed_matrix_mult(A, B)
    end_dist = time.perf_counter()
    tempo_dist = end_dist - start_dist
    print(f"Tempo distribuído: {tempo_dist:.6f} segundos\n")

    # -----------------------------
    # Validação dos resultados
    # -----------------------------
    iguais = np.array_equal(C_serial, C_dist)
    print(">>> VALIDAÇÃO DOS RESULTADOS")
    print(f"As matrizes resultantes (serial e distribuída) são iguais? {iguais}")
    print("\nResumo:")
    print(f"  - Tempo Serial      : {tempo_serial:.6f} s")
    print(f"  - Tempo Distribuído : {tempo_dist:.6f} s")

    if tempo_dist < tempo_serial:
        print("  -> A versão distribuída foi mais rápida para esse tamanho de matriz.")
    else:
        print("  -> A versão distribuída NÃO foi mais rápida (overhead de paralelismo/ comunicação).")

    # -----------------------------
    # Tabela de Resultados
    # -----------------------------
    print("\n=======================================")
    print("           Tabela de Resultados          ")
    print("=======================================\n")

    print(f"{'Tamanho (NxN)':<20} {'Serial (s)':<15} {'Distribuído (s)':<18} {'Igual?':<10} {'Mais rápida':<20}")
    print("-" * 85)
    resultado = "Serial" if tempo_serial < tempo_dist else "Distribuída"
    print(f"{N:<20} {tempo_serial:<15.6f} {tempo_dist:<18.6f} {iguais}       {resultado}")

    print("\nFim da execução.")

if __name__ == "__main__":
    main()
