import socket
import pickle
import numpy as np

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

def multiply(subA, B):
    return np.dot(subA, B)

def start_server():
    HOST = "127.0.0.1"
    PORT = 5002  # Porta do servidor 2

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print(f"[SERVER 2] Servidor iniciado em {HOST}:{PORT}, aguardando conexões...")

    while True:
        conn, addr = server_socket.accept()
        print(f"[SERVER 2] Conectado a {addr}")

        try:
            subA, B = recv_pickle(conn)
            print(f"[SERVER 2] Recebeu submatriz A com shape {subA.shape} e B com shape {B.shape}")

            result = multiply(subA, B)
            print(f"[SERVER 2] Multiplicação concluída. Enviando resultado com shape {result.shape}...")

            send_pickle(conn, result)
            print("[SERVER 2] Resultado enviado com sucesso.\n")

        except Exception as e:
            print(f"[SERVER 2] Erro durante o processamento: {e}")

        finally:
            conn.close()

if __name__ == "__main__":
    start_server()
