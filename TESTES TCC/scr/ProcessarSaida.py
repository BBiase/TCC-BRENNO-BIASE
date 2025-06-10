import pandas as pd

modelo = None
resultados = {}
with open('saida.txt', 'r', encoding='Cp1252') as f:
    linhas = f.readlines()
    for linha in linhas:
        linha = linha.strip()
        if linha.startswith('='):
            modelo = linha.replace('=', '').strip()
            resultados[modelo] = {}
            flag = 0
        elif linha.startswith("TC"):
            flag += 1
            resultados[modelo][flag] = {}
            resultados[modelo][flag]["TC"] = linha.split(":")[1].strip().replace('.', ',')
        elif linha.startswith("PE"):
            aux = linha.split('-')[0].split(':')
            resultados[modelo][flag]["PE"] = aux[1].strip()
        elif linha.startswith("TR") or linha.startswith("CL"):
            aux = linha.split(':')
            resultados[modelo][flag][linha[:2]] = aux[1].strip().replace('.',',')
tabela = []
for modelo, v1 in resultados.items():
    for ordem, v2 in v1.items():
        t = {}
        t["MD"] = modelo
        t["TC"] = v2["TC"]
        t["PE"] = v2["PE"]
        t["TR"] = v2["TR"]
        t["CL"] = v2["CL"]
        tabela.append(t)

for linha in tabela:
    print(linha)
    
df = pd.DataFrame(tabela)
df.to_excel('saida1.xlsx')