# Importa o módulo ollama (para interagir com o modelo de linguagem local)
# e a função time (para medir o tempo de execução)
import ollama
from time import time

# Inicia a contagem de tempo para medir quanto o modelo leva para responder pela primeira vez
tempo_carregamento = time()
# Envia um prompt de teste ao modelo para ativá-lo/carregá-lo
resultado = ollama.generate(model="deepseek-r1:32b", prompt="Bom dia")
# Calcula o tempo de carregamento da primeira resposta
tempo_carregamento = time() - tempo_carregamento

# Lê o conteúdo do arquivo que contém o regulamento (contexto para as perguntas)
contexto = open('regulamento_regular.txt', encoding='utf-8').readlines()
# Junta todas as linhas em uma única string (removendo quebras de linha)
contexto = "".join(contexto)

# Lista de perguntas que podem ser selecionadas para serem respondidas pelo modelo
perguntas = [
    '1-O que significa TCC?',
    '2-Quais são os objetivos da monografia?',
    '3-Quais são as atribuições do coordenador?',
    '4-O que o aluno deve fazer se não encontrar um orientador?',
    '5-Quem deve escrever a monografia?',
    '6-Quanto o aluno deve tirar para ser aprovado no TCC?',
    '7-O aluno que tirar nota 7 está aprovado?',
    '8-Quantos orientandos cada orientador pode ter?' 
]

# Loop principal: repete até o ser digitar "SAIR" ou deixar a entrada em branco
while True:
    print("Perguntas:\n")
    # Exibe as perguntas disponíveis
    for pergunta in perguntas:
        print(pergunta)

    # Solicita o número da pergunta desejada
    pergunta_original = input("Digite o código da pergunta (ou SAIR para finalizar): ")
    
    # Se o usuário digitar "SAIR" ou apenas apertar Enter sem digitar nada, encerra o programa
    if pergunta_original.upper() == "SAIR" or pergunta_original.strip() == "":
        break

    # Converte o código numérico em índice e recupera a pergunta da lista
    pergunta_original = int(pergunta_original)
    pergunta_original = perguntas[pergunta_original-1]

    # Cria o prompt completo para o modelo, combinando contexto e pergunta
    pergunta = f"Considere o contexto abaixo e responda em português a pergunta a seguir:\n\nContexto:\n{contexto}\n\nPergunta: {pergunta_original}"

    # Mede o tempo que o modelo leva para responder essa pergunta
    tempo_resposta = time()
    resultado = ollama.generate(model="deepseek-r1:32b", prompt=pergunta) 
    tempo_resposta = time() - tempo_resposta

    # Extrai o texto da resposta do resultado
    resposta = resultado["response"]

    # Exibe a resposta recebida do modelo
    print("\n\n=============================================================\n\n")
    print(resposta)  # <- Aqui a resposta é exibida de uma vez só
    print("\n\n=============================================================\n\n")

    print("Classificações:\n5-Completamente correto\n4-Correto (prolixo ou português errado)\n3-Parcialmente correto\n2-Desvio do contexto\n1-Completamente equivocado (alucinação)\n0-Predominantemente, em outro idioma\n\n")
    
    # inserir a o peso de classificação da resposta
    classificacao = input("Qual a qualidade da resposta recebida: ")

    # Abre (ou cria) o arquivo saida.txt e grava os dados da execução atual
    with open('saida.txt', 'a', encoding="utf-8") as arquivo:
        arquivo.write(f"TC:{tempo_carregamento}\n")                     # Tempo de carregamento
        arquivo.write(f"PE:{pergunta_original}\n")                      # Pergunta feita
        arquivo.write(f"TR:{tempo_resposta}\n")                         # Tempo de resposta
        arquivo.write(f"RE:{resposta.replace('\n', ' ')}\n")            # Resposta (em linha única)
        arquivo.write(f"CL:{classificacao}\n\n")                        # Classificação dada

'''
A cada pergunta respondida, serão salvos no arquivo saida.txt:
- TC: Tempo de carregamento do modelo nesta execução
- PE: A pergunta feita
- TR: O tempo de resposta dessa pergunta
- RE: A resposta
- CL: A classificação dada a essa resposta
'''