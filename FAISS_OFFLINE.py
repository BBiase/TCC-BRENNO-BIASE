import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from time import time
import os

# -----------------------------
# Configurações
# -----------------------------
log_file = "resultados_faiss_offline.txt"
dados_file = "regulamento_regular.txt"
modelo_nome = "DeepSeek-r1:7b"

# -----------------------------
# Contador de perguntas
# -----------------------------
contador_pe = 1

# -----------------------------
# Carregar modelo local de embeddings (offline)
# -----------------------------
tempo_inicio = time()
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # modelo mais preciso
tempo_fim = time()
tc_modelo = tempo_fim - tempo_inicio

print("Bom dia! O modelo de embeddings foi carregado com sucesso!")
print(f"TC: {tc_modelo:.6f} s\n")

# -----------------------------
# Função para gerar embeddings
# -----------------------------
def gerar_embedding(texto):
    return model.encode(texto)

# -----------------------------
# Função para carregar/criar índice FAISS
# -----------------------------
def carregar_indice(dados, index_file="index.faiss"):
    if not os.path.exists(index_file):
        embeddings = np.array([gerar_embedding(texto) for texto in dados]).astype('float32')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_file)
        print("Índice FAISS criado e salvo.")
    else:
        index = faiss.read_index(index_file)
        print("Índice FAISS carregado do disco.")
    return index

# -----------------------------
# Função para buscar resposta
# -----------------------------
def buscar_resposta(pergunta, dados, index, qtd=5):
    embedding_pergunta = gerar_embedding(pergunta)
    embedding_pergunta = np.array(embedding_pergunta, dtype='float32').reshape(1, -1)
    dist, indices = index.search(embedding_pergunta, qtd)
    respostas = [dados[indices[0][i]] for i in range(len(indices[0]))]
    return " ".join(respostas)  # resposta em linha única

# -----------------------------
# Carregar dados do arquivo
# -----------------------------
with open(dados_file, 'r', encoding='utf-8') as f:
    dados = [linha.strip() for linha in f.readlines() if linha.strip()]

# -----------------------------
# Criar ou carregar índice FAISS
# -----------------------------
index = carregar_indice(dados)

# -----------------------------
# Perguntas de teste
# -----------------------------
perguntas = [
    "O que significa TCC?",    
    "Quais são os objetivos da monografia?",
    "Quais são as atribuições do coordenador?",
    "O que o aluno deve fazer se não encontrar um orientador?",
    "Quem deve escrever a monografia?",
    "Quanto o aluno deve tirar para ser aprovado em TCC?",
    "O aluno que tirar nota 7 está aprovado?",
    "Quantos orientandos cada orientador pode ter?"
]

# -----------------------------
# Executar perguntas
# -----------------------------
for pergunta_usuario in perguntas:
    print(f"PE: {pergunta_usuario}")

    tempo_resposta_inicio = time()
    resposta = buscar_resposta(pergunta_usuario, dados, index, qtd=5)
    tempo_resposta = time() - tempo_resposta_inicio

    # Mostrar resposta antes de solicitar nota
    print(f"RE: {resposta}")
    print(f"TR: {tempo_resposta:.6f} s")

    # Solicitar nota geral
    while True:
        try:
            nota = int(input("Atribua uma nota geral de 0 a 5 para esta pergunta: "))
            if 0 <= nota <= 5:
                break
            else:
                print("Nota deve ser entre 0 e 5.")
        except ValueError:
            print("Digite um número inteiro entre 0 e 5.")

    print(f"CL: {nota}\n")

    # Salvar no arquivo no padrão desejado
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"TC:{tc_modelo:.6f}\n")
        f.write(f"PE:{contador_pe}-{pergunta_usuario}\n")
        f.write(f"TR:{tempo_resposta:.6f}\n")
        f.write(f"RE:{resposta}\n")
        f.write(f"CL:{nota}\n\n")

    contador_pe += 1
