import socket
import pickle
import numpy as np

# ----------------------------------------------------------
# Funções auxiliares para enviar e receber objetos via socket
# ----------------------------------------------------------

def receber_bytes(sock, n_bytes):
    """Recebe exatamente n_bytes do socket."""
    dados = b""
    while len(dados) < n_bytes:
        pacote = sock.recv(n_bytes - len(dados))
        if not pacote:
            break
        dados += pacote
    return dados

def receber_pickle(sock):
    """Recebe objeto Python serializado (pickle) via socket."""
    cabecalho = receber_bytes(sock, 8)
    tamanho = int.from_bytes(cabecalho, "big")
    dados = receber_bytes(sock, tamanho)
    return pickle.loads(dados)

def enviar_pickle(sock, obj):
    """Envia objeto Python via socket usando pickle."""
    dados = pickle.dumps(obj)
    cabecalho = len(dados).to_bytes(8, "big")
    sock.sendall(cabecalho + dados)

# ----------------------------------------------------------
# Função principal de multiplicação
# ----------------------------------------------------------

def multiplicar(A, B):
    """Realiza a multiplicação de matrizes."""
    return np.dot(A, B)

# ----------------------------------------------------------
# Servidor principal
# ----------------------------------------------------------

def iniciar_servidor():
    HOST = "127.0.0.1"
    PORTA = 5001

    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.bind((HOST, PORTA))
    servidor.listen()

    print(f"[SERVIDOR 1] Servidor aguardando conexões na porta {PORTA}...")

    while True:
        conexao, endereco = servidor.accept()
        print(f"[SERVIDOR 1] Conectado a {endereco}")

        # Recebe A1, B1 e B2
        A_parte, B1, B2 = receber_pickle(conexao)

        print("[SERVIDOR 1] Efetuando multiplicações: C11 = A1×B1 e C12 = A1×B2")
        C11 = multiplicar(A_parte, B1)
        C12 = multiplicar(A_parte, B2)

        # Envia os blocos C11 e C12
        enviar_pickle(conexao, (C11, C12))
        conexao.close()

if __name__ == "__main__":
    iniciar_servidor()
