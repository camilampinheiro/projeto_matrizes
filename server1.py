import socket
import pickle
import numpy as np

# ==========================
# Funções auxiliares
# ==========================

def recv_all(sock, n_bytes):
    """Lê exatamente n_bytes do socket."""
    data = b""
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            raise ConnectionError("Conexão interrompida durante o recebimento.")
        data += packet
    return data

def recv_pickle(sock):
    """Recebe um objeto Python serializado com pickle (com cabeçalho de tamanho)."""
    # Primeiro, lê 8 bytes com o tamanho
    header = recv_all(sock, 8)
    msg_len = int.from_bytes(header, byteorder="big")
    # Depois, lê a mensagem completa
    data = recv_all(sock, msg_len)
    return pickle.loads(data)

def send_pickle(sock, obj):
    """Envia um objeto Python via socket usando pickle com cabeçalho de tamanho."""
    data = pickle.dumps(obj)
    header = len(data).to_bytes(8, byteorder="big")
    sock.sendall(header + data)

# ==========================
# Lógica do servidor
# ==========================

def multiply(subA, B):
    """Multiplica a submatriz A (subA) pela matriz B."""
    return np.dot(subA, B)

def start_server():
    HOST = "127.0.0.1"
    PORT = 5001  # Porta do servidor 1

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print(f"[SERVER 1] Servidor iniciado em {HOST}:{PORT}, aguardando conexões...")

    while True:
        conn, addr = server_socket.accept()
        print(f"[SERVER 1] Conectado a {addr}")

        try:
            # Recebe submatriz de A e matriz B
            subA, B = recv_pickle(conn)
            print(f"[SERVER 1] Recebeu submatriz A com shape {subA.shape} e B com shape {B.shape}")

            # Faz a multiplicação
            result = multiply(subA, B)
            print(f"[SERVER 1] Multiplicação concluída. Enviando resultado com shape {result.shape}...")

            # Envia resultado de volta ao cliente
            send_pickle(conn, result)
            print("[SERVER 1] Resultado enviado com sucesso.\n")

        except Exception as e:
            print(f"[SERVER 1] Erro durante o processamento: {e}")

        finally:
            conn.close()

if __name__ == "__main__":
    start_server()