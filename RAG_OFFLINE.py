# RAG_FAISS_OLLAMA_Interactive.py
# Pipeline RAG: FAISS (recuperação offline) + Ollama (geração)
# Salva Saida_RAG.txt com: TC, PE, RE, TR, CL
# Salva Contextos_RAG.txt apenas uma vez

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from time import time
import ollama

# ----------------------------
# CONFIGURAÇÕES - edite aqui
# ----------------------------
ARQUIVO_REGULAMENTO = "regulamento_regular.txt"
ARQUIVO_INDICE = "index.faiss"
ARQUIVO_SAIDA = "Saida_RAG.txt"
ARQUIVO_CONTEXTO = "Contextos_RAG.txt"

EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
TOP_K = 5  

# Modelo Ollama (alterar manualmente conforme teste)
OLLAMA_MODEL = "deepseek-r1:8b"

# Perguntas
PERGUNTAS = [
    "O que significa TCC?",
    "Quais são os objetivos da monografia?",
    "Quais são as atribuições do coordenador?",
    "O que o aluno deve fazer se não encontrar um orientador?",
    "Quem deve escrever a monografia?",
    "Quanto o aluno deve tirar para ser aprovado em TCC?",
    "O aluno que tirar nota 7 está aprovado?",
    "Quantos orientandos cada orientador pode ter?"
]

# ----------------------------
# Carregar regulamento (dados)
# ----------------------------
if not os.path.exists(ARQUIVO_REGULAMENTO):
    raise FileNotFoundError(f"Arquivo de regulamento não encontrado: {ARQUIVO_REGULAMENTO}")

with open(ARQUIVO_REGULAMENTO, "r", encoding="utf-8") as f:
    dados = [linha.strip() for linha in f.readlines() if linha.strip()]

# ----------------------------
# Carregar modelo de embeddings + FAISS
# ----------------------------
print("Carregando modelo de embeddings (offline)...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def gerar_embedding(texto: str) -> np.ndarray:
    return embed_model.encode(texto)

def carregar_ou_criar_indice(textos, index_file=ARQUIVO_INDICE):
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print("Índice FAISS carregado do disco.")
        return index
    print("Criando índice FAISS (pode demorar na primeira execução)...")
    embs = np.array([gerar_embedding(t) for t in textos]).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, index_file)
    print("Índice FAISS criado e salvo.")
    return index

index = carregar_ou_criar_indice(dados, ARQUIVO_INDICE)

def recuperar_contexto(pergunta: str, top_k: int = TOP_K) -> str:
    q_emb = np.array(gerar_embedding(pergunta), dtype="float32").reshape(1, -1)
    dist, idxs = index.search(q_emb, top_k)
    passagens = [dados[idx] for idx in idxs[0]]
    return "\n".join(passagens)

# ----------------------------
# Salvar contexto FAISS apenas uma vez
# ----------------------------
if not os.path.exists(ARQUIVO_CONTEXTO):
    print("Salvando contexto FAISS para acompanhamento (uma única vez)...")
    with open(ARQUIVO_CONTEXTO, "w", encoding="utf-8") as f_ctx:
        f_ctx.write("="*30 + OLLAMA_MODEL + "="*30 + "\n")
        for pergunta in PERGUNTAS:
            contexto = recuperar_contexto(pergunta, TOP_K)
            f_ctx.write(f"PE:{pergunta}\n")
            f_ctx.write("CTX_BEGIN\n")
            f_ctx.write(contexto.replace("\n", " ") + "\n")
            f_ctx.write("CTX_END\n\n")
    print(f"Contextos salvos em: {ARQUIVO_CONTEXTO}")
else:
    print(f"Arquivo de contexto já existe: {ARQUIVO_CONTEXTO} — não será sobrescrito.")

# ----------------------------
# TC do modelo Ollama
# ----------------------------
print("\nAquecendo o modelo Ollama para medir TC (warmup)...")
t0 = time()
try:
    _ = ollama.generate(model=OLLAMA_MODEL, prompt="Bom dia")
except Exception as e:
    print("Aviso: erro durante carregamento do Ollama.", e)
TC = time() - t0
print(f"TC (warmup LLM): {TC:.6f}s\n")

# ----------------------------
# Função para responder via Ollama
# ----------------------------
def responder_com_llm(contexto_filtrado: str, pergunta: str) -> (str, float):
    prompt = (
        f"Considere o contexto abaixo e responda em português a pergunta a seguir:\n\n"
        f"Contexto:\n{contexto_filtrado}\n\n"
        f"Pergunta: {pergunta}"
    )
    t0 = time()
    resultado = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    TR = time() - t0
    resposta = resultado.get("response", "").strip()
    return resposta, TR

# ----------------------------
# Execução RAG interativa
# ----------------------------
print("Iniciando experimento RAG (FAISS -> LLM).")

# Flag para escrever a linha de delimitação do modelo apenas uma vez
modelo_escrito = False

with open(ARQUIVO_SAIDA, "a", encoding="utf-8") as out:

    while True:
        # Exibe apenas o menu de perguntas
        print("\nPerguntas disponíveis (digite o número correspondente ou SAIR para finalizar):\n")
        for i, pergunta in enumerate(PERGUNTAS, 1):
            print(f"{i} - {pergunta}")

        escolha = input("\nDigite o número da pergunta desejada (ou SAIR): ").strip()
        if escolha.upper() == "SAIR" or escolha == "":
            break

        try:
            idx = int(escolha) - 1
            if idx < 0 or idx >= len(PERGUNTAS):
                print("Número inválido, tente novamente.")
                continue
        except ValueError:
            print("Entrada inválida, tente novamente.")
            continue

        pergunta_selecionada = PERGUNTAS[idx]
        contexto_filtrado = recuperar_contexto(pergunta_selecionada, TOP_K)

        resposta, TR = responder_com_llm(contexto_filtrado, pergunta_selecionada)

        print("\n--- RESPOSTA (LLM) ---")
        print(resposta)
        print(f"\nTR_LLM: {TR:.6f}s\n")
        print("-"*80 + "\n")

        print("Classificações:\n"
              "5 - Completamente correto\n"
              "4 - Correto (prolixo ou português errado)\n"
              "3 - Parcialmente correto\n"
              "2 - Desvio do contexto\n"
              "1 - Completamente equivocado (alucinação)\n"
              "0 - Predominantemente, em outro idioma\n")

        classificacao = input("Qual a qualidade da resposta recebida (0-5, ou Enter para pular): ").strip()
        if classificacao == "":
            classificacao = "NA"

        # Escreve linha de modelo apenas uma vez por execução
        if not modelo_escrito:
            out.write("="*30 + OLLAMA_MODEL + "="*30 + "\n")
            modelo_escrito = True

        # Salvamento
        out.write(f"TC:{TC:.6f}\n")
        out.write(f"PE:{pergunta_selecionada}\n")
        out.write(f"RE:{resposta.replace(chr(10), ' ')}\n")
        out.write(f"TR:{TR:.6f}\n")
        out.write(f"CL:{classificacao}\n\n")
        out.flush()

print(f"\nExecução finalizada. Resultados anexados em: {ARQUIVO_SAIDA}")
