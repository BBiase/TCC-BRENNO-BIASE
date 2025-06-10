import faiss
import numpy as np
import os
import openai
from time import time

os.environ['OPENAI_API_KEY'] = "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"


def carregar_indice(dados, index_file="index.faiss"):
    index = None
    if not os.path.exists(index_file):
        embeddings = np.array([gerar_embedding(texto) for texto in dados])
        # criar indice
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print("Modelo criado com sucesso!")
        # salvar indice
        faiss.write_index(index, index_file)
        print("Modelo salvo com sucesso! FAISS")
    else:
        index = faiss.read_index(index_file)
        print("Modelo carregado com sucesso! FAISS")
    return index
    

def gerar_embedding(texto):
    response = openai.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def buscar_resposta(pergunta, dados, index, threshold=1.0, qtd = 1):
    # Criar embedding para a pergunta
    embedding_pergunta = np.array([gerar_embedding(pergunta)])
    
    # Buscar no FAISS o texto mais próximo
    dist, indices = index.search(embedding_pergunta, qtd)
    if dist[0][0] > threshold:
        return "Tente perguntar de outra forma."
    # Retornar a resposta mais relevante
    resposta_relevante = ""
    for i in range(len(indices[0])):
        resposta_relevante += dados[indices[0][i]] + "\n"
    
    return resposta_relevante

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
with open('regulamento_regular.txt', 'r', encoding='utf-8') as f:
    dados = f.readlines()
tempo_carregamento = time()
index = carregar_indice(dados)
tempo_carregamento = time() - tempo_carregamento

for ix, pergunta_usuario in enumerate(perguntas):
    print("TC:" + str(tempo_carregamento))
    print("PE:" + pergunta_usuario)

    tempo_resposta = time()
    ctx = buscar_resposta(pergunta_usuario, dados, index)
    tempo_resposta = time() - tempo_resposta
    
    print("TR:" + str(tempo_resposta))
    
    print("RE:" + ctx)