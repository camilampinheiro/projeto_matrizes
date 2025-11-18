import numpy as np
import time
import socket
import threading
import pickle

# ----------------------------------------------------------
# Funções auxiliares de comunicação
# ----------------------------------------------------------

def receber_bytes(sock, n_bytes):
    dados = b""
    while len(dados) < n_bytes:
        pacote = sock.recv(n_bytes - len(dados))
        if not pacote:
            break
        dados += pacote
    return dados

def receber_pickle(sock):
    cabecalho = receber_bytes(sock, 8)
    tamanho = int.from_bytes(cabecalho, "big")
    dados = receber_bytes(sock, tamanho)
    return pickle.loads(dados)

def enviar_pickle(sock, obj):
    dados = pickle.dumps(obj)
    cabecalho = len(dados).to_bytes(8, "big")
    sock.sendall(cabecalho + dados)

# ----------------------------------------------------------
# Execução Serial
# ----------------------------------------------------------

def multiplicacao_serial(A, B):
    """Multiplicação tradicional (sem paralelismo)."""
    return np.dot(A, B)

# ----------------------------------------------------------
# Execução Paralela Local (duas threads)
# ----------------------------------------------------------

def thread_worker(A_parte, B, resultados, indice):
    """Thread que calcula parte da matriz."""
    resultados[indice] = np.dot(A_parte, B)

def multiplicacao_paralela(A, B):
    """Divide A em duas partes e multiplica em paralelo local."""
    A1, A2 = np.array_split(A, 2, axis=0)
    resultados = {}
    
    t1 = threading.Thread(target=thread_worker, args=(A1, B, resultados, 0))
    t2 = threading.Thread(target=thread_worker, args=(A2, B, resultados, 1))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    return np.vstack((resultados[0], resultados[1]))

# ----------------------------------------------------------
# Execução Distribuída (2 servidores)
# ----------------------------------------------------------

def worker_distribuido(host, porta, A_parte, B1, B2, resultados, indice):
    """Envia partes da matriz para o servidor e recebe blocos C."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, porta))
    
    enviar_pickle(s, (A_parte, B1, B2))
    resposta = receber_pickle(s)
    s.close()
    
    resultados[indice] = resposta

def multiplicacao_distribuida(A, B):
    """Divide A e B corretamente e coordena os dois servidores."""
    
    A1, A2 = np.array_split(A, 2, axis=0)
    B1, B2 = np.array_split(B, 2, axis=1)
    
    resultados = {}

    t1 = threading.Thread(target=worker_distribuido, args=("127.0.0.1", 5001, A1, B1, B2, resultados, 0))
    t2 = threading.Thread(target=worker_distribuido, args=("127.0.0.1", 5002, A2, B1, B2, resultados, 1))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    C11, C12 = resultados[0]
    C21, C22 = resultados[1]

    superior = np.hstack((C11, C12))
    inferior = np.hstack((C21, C22))
    C_final = np.vstack((superior, inferior))

    return C_final

# ----------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------

def main():
    N = 1200  # tamanho da matriz
    np.random.seed(0)

    A = np.random.randint(0, 10, (N, N))
    B = np.random.randint(0, 10, (N, N))

    print("\n================ EXECUÇÃO SERIAL ================")
    t0 = time.time()
    C_serial = multiplicacao_serial(A, B)
    t1 = time.time()
    tempo_serial = t1 - t0
    print(f"Tempo serial: {tempo_serial:.6f}s")

    print("\n================ EXECUÇÃO PARALELA LOCAL ================")
    t0 = time.time()
    C_paralelo = multiplicacao_paralela(A, B)
    t1 = time.time()
    tempo_paralelo = t1 - t0
    print(f"Tempo paralelo local: {tempo_paralelo:.6f}s")

    print("\n================ EXECUÇÃO DISTRIBUÍDA ================")
    t0 = time.time()
    C_dist = multiplicacao_distribuida(A, B)
    t1 = time.time()
    tempo_dist = t1 - t0
    print(f"Tempo distribuído: {tempo_dist:.6f}s")

    print("\n================ VALIDAÇÃO DOS RESULTADOS ================")
    """
    Essa seção serve para comprovar que a implementação paralela e a distribuída estão:
    ✔ Produzindo exatamente o mesmo resultado que a versão serial
    ✔ Ou seja: não houve erros na matemática
    ✔ Não houve erro no envio/recebimento de matrizes
    ✔ Não houve erro na recomposição da matriz final
    """
    iguais_paralelo = np.array_equal(C_serial, C_paralelo)
    iguais_distribuido = np.array_equal(C_serial, C_dist)

    print(f"Serial == Paralelo Local : {iguais_paralelo}")
    print(f"Serial == Distribuído    : {iguais_distribuido}")

    # -------- Cálculo de Speedup e Eficiência --------
    speedup_paralelo = tempo_serial / tempo_paralelo
    speedup_distribuido = tempo_serial / tempo_dist

    eficiencia_paralelo = speedup_paralelo / 2
    eficiencia_distribuido = speedup_distribuido / 2

    print("\n================ TABELA DETALHADA DE RESULTADOS ================")
    print(f"{'Método':<25} {'Tempo (s)':<12} {'Speedup':<12} {'Eficiência':<12}")
    print("-" * 65)
    print(f"{'Serial':<25} {tempo_serial:<12.6f} {'-':<12} {'-':<12}")
    print(f"{'Paralelo Local':<25} {tempo_paralelo:<12.6f} {speedup_paralelo:<12.3f} {eficiencia_paralelo:<12.3f}")
    print(f"{'Distribuído (2 serv.)':<25} {tempo_dist:<12.6f} {speedup_distribuido:<12.3f} {eficiencia_distribuido:<12.3f}")

    print("\n================ ANÁLISE AUTOMÁTICA ================")

    melhor = min([
        ('Serial', tempo_serial),
        ('Paralelo Local', tempo_paralelo),
        ('Distribuído', tempo_dist)
    ], key=lambda x: x[1])

    print(f"- Método mais rápido: {melhor[0]} ({melhor[1]:.6f}s)\n")

    print("• Speedup Paralelo Local :", f"{speedup_paralelo:.2f}x")
    print("• Speedup Distribuído     :", f"{speedup_distribuido:.2f}x")

    print("\n• Eficiência Paralelo Local :", f"{eficiencia_paralelo*100:.1f}%")
    print("• Eficiência Distribuído     :", f"{eficiencia_distribuido*100:.1f}%")

    print("\n================ FIM DA EXECUÇÃO ================")

if __name__ == "__main__":
    main()
