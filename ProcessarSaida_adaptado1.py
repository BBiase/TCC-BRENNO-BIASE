import pandas as pd

# Inicialização
modelo = None
resultados = {}
flag = 0

# Abrir arquivo de resultados
with open('SAIDA_RAG.txt', 'r', encoding='utf-8') as f:
    linhas = f.readlines()
    for linha in linhas:
        linha = linha.strip()
        if linha.startswith('='):
            # Captura o modelo
            modelo = linha.replace('=', '').strip()
            resultados[modelo] = {}
            flag = 0
        elif linha.startswith("TC"):
            # Novo bloco
            flag += 1
            resultados[modelo][flag] = {}
            resultados[modelo][flag]["TC"] = linha.split(":", 1)[1].strip().replace('.', ',')
        elif linha.startswith("PE"):
            # Garante que o dicionário do bloco existe
            if flag not in resultados[modelo]:
                resultados[modelo][flag] = {}
            # Pega tudo após o "-"
            aux = linha.split(":")[-1]
            resultados[modelo][flag]["PE"] = aux.strip()
        elif linha.startswith("RE"):
            # Garante que o dicionário do bloco existe
            if flag not in resultados[modelo]:
                resultados[modelo][flag] = {}
            # Pega tudo após o "-"
            aux = linha.split("RE:")[-1]
            if '</think>' in aux.lower():
                aux = aux.split("</think>")[-1]
            resultados[modelo][flag]["RE"] = aux.strip()
        elif linha.startswith("TR") or linha.startswith("CL"):
            aux = linha.split(":", 1)
            resultados[modelo][flag][linha[:2]] = aux[1].strip().replace('.',',')

# Transformar em tabela
tabela = []
for modelo, v1 in resultados.items():
    for ordem, v2 in v1.items():
        t = {}
        t["MD"] = modelo
        t["TC"] = v2.get("TC","")
        t["PE"] = v2.get("PE","")
        t["RE"] = v2.get("RE","")
        t["TR"] = v2.get("TR","")
        t["CL"] = v2.get("CL","")
        tabela.append(t)

# Criar DataFrame e exportar para Excel
df = pd.DataFrame(tabela)
df = df[["MD", "PE", "RE"]]
df = df.sort_values(by=['PE', 'MD'])
# df.to_excel('saida_exemplo.xlsx', index=False)

aux = {}
for index, row in df.iterrows():
    modelo = row['MD']
    pergunta = row['PE']
    resposta = row['RE']
    x = aux.get(pergunta

# Opcional: mostrar no console
print(df)
