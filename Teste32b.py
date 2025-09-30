# Importa o módulo ollama (para interagir com o modelo de linguagem local)
# e a função time (para medir o tempo de execução)
import ollama
from time import time

# Nome do modelo a ser testado
modelo = "Qwen:32b"

# Inicia a contagem de tempo para medir quanto o modelo leva para responder pela primeira vez (carregamento)
tempo_carregamento = time()
# Envia um prompt de teste ao modelo para ativá-lo/carregá-lo
resultado = ollama.generate(model=modelo, prompt="Teste inicial para carregar o modelo.")
# Calcula o tempo de carregamento da primeira resposta
tempo_carregamento = time() - tempo_carregamento

# Lê o conteúdo do arquivo que contém o regulamento (contexto para as perguntas)
contexto = open('regulamento_regular.txt', encoding='utf-8').readlines()
contexto = "".join(contexto)

# Duas perguntas fixas para o teste
perguntas = [
    '1-O que significa TCC?'
]

# Arquivo de saída
with open('saida.txt', 'a', encoding="utf-8") as arquivo:
    # Grava o tempo de carregamento do modelo
    arquivo.write(f"Modelo: {modelo}\n")
    arquivo.write(f"TC:{tempo_carregamento}\n")

    # Loop automático nas perguntas
    for pergunta_original in perguntas:
        # Monta o prompt com contexto
        pergunta = f"Considere o contexto abaixo e responda em português a pergunta a seguir:\n\nContexto:\n{contexto}\n\nPergunta: {pergunta_original}"

        # Mede o tempo de resposta
        tempo_resposta = time()
        resultado = ollama.generate(model=modelo, prompt=pergunta)
        tempo_resposta = time() - tempo_resposta

        # Extrai a resposta
        resposta = resultado["response"]

        # Grava no arquivo
        arquivo.write(f"PE:{pergunta_original}\n")                  # Pergunta
        arquivo.write(f"TR:{tempo_resposta}\n")                     # Tempo de resposta
        arquivo.write(f"RE:{resposta.replace('\n', ' ')}\n\n")      # Resposta (em linha única)

print("Teste concluído. Resultados salvos em 'saida.txt'")