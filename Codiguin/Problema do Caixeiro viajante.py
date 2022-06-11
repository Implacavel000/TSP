# TODO: Pesquisar Quantum Annealing
from ast import If
import qiskit
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def dimMatriz(matriz):
    qtdLins = len(matriz)
    if qtdLins == 0:
        return 0, 0
    else:
        qtdCols = len(matriz[0])
        return qtdLins, qtdCols


n = 4

"""Adicionar nós"""
G = nx.DiGraph()

for i in range(1, n+1):
    G.add_node(i)


"""Adicionar ligações"""
for i in range(1, n+1):
    for j in range(1, n+1):
        if j!=i:
            G.add_edge(i,j)


"""Posição dos nós"""
pos = None
list1 = [[None]]
for i in range(1, n):
    arlist = []

    l = random.randrange(-100, 100, 25)/100
    m = random.randrange(-100, 100, 25)/100
    list = [l, m]
    if list1 == [[None]]:
        list1 = [list]
    qtdLins, qtdCols = dimMatriz(list1)

    for x in range(qtdLins):
        if list[0] == list1[x][0] and list[1] == list1[x][1]:
            l = random.randrange(-100, 100, 25)/100
            m = random.randrange(-100, 100, 25)/100
            list = [l, m]

    else:
        list1 = list1 + [list]

    # TODO: Censertar os vetores posições 
    #print(pos1)

print(list1)

for i in range(n):
    if pos == None:
        pos = {
            i+1: list1[i]
            }
        print(pos)
    else:
        pos[i+1] = list1[i]

edge_labels = {}
'''# Parâmetros testes
edge_labels = {}
pos = {1: [0.75, 1.0],
      2: [0.75, 0.15],
      3: [0.5, -0.5],
      4: [1.0, -0.5]}
'''
"""Etiquetas do grafo"""

edge_labels = None
labels5 = ""


for i in range(1, n):
    for j in range(1, n+1): 
        if j > i:
            edge_labels = {
                (i, j): '$\\phi_{',j,'\\to ',i,'}$\n $\\phi_{',i,'\\to ',j,'}$'
            }





            labels = '$\\phi_{',j,'\\to ',i,'}$\n $\\phi_{',i,'\\to ',j,'}$'
            print(labels)
            labels1 = str(labels)
            labels2 = labels1.replace(", ", "").replace("'", "").replace("(", "").replace(")", "")
            labels3 = f" '{labels2}',"
            print(labels3)
            labels4 = f"({i}, {j}):{labels3}"
            labels5 = labels5 + "\n" + labels4
            labels6 = labels5[:-1] + "\n"
            edge_labels = {
            (i, j): labels3
            }




#print(labels6)





fig = plt.figure(1, figsize=(14, 10)) 
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos, connectionstyle='arc3, rad = 0.2', 
        node_size=3000, arrowsize=14, arrowstyle='simple', font_size=30)

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20, bbox=dict(alpha=0))
 
plt.show()


# Matriz =
# Inserir informação (Matriz de Matriz) das distancias entre as cidades pelos n caminhos
# TODO: resolver o problema matriz de matrizes
# TODO: Usar particulas ou trafego cotidiano??
# TODO: Receber informação da velocidade media de cada rota (De preferencia por horario do dia e dia da semana) COMO????
# TODO: Se possivel com dados atualizados (GPS? Waze? com data, hora e velocidade do trafego)