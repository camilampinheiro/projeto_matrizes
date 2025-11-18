import socket
import pickle
import numpy as np

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

def multiplicar(A, B):
    return np.dot(A, B)

def iniciar_servidor():
    HOST = "127.0.0.1"
    PORTA = 5002

    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.bind((HOST, PORTA))
    servidor.listen()

    print(f"[SERVIDOR 2] Servidor aguardando conexões na porta {PORTA}...")

    while True:
        conexao, endereco = servidor.accept()
        print(f"[SERVIDOR 2] Conectado a {endereco}")

        # Recebe A2, B1 e B2
        A_parte, B1, B2 = receber_pickle(conexao)

        print("[SERVIDOR 2] Efetuando multiplicações: C21 = A2×B1 e C22 = A2×B2")
        C21 = multiplicar(A_parte, B1)
        C22 = multiplicar(A_parte, B2)

        enviar_pickle(conexao, (C21, C22))
        conexao.close()

if __name__ == "__main__":
    iniciar_servidor()
