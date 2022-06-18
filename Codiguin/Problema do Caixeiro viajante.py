# TODO: Pesquisar Quantum Annealing
from ast import If
from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister, execute
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, array_to_latex
from qiskit.circuit.library import QFT
from numpy import pi
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qiskit.tools.jupyter

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


# Parâmetro posição teste
pos = {1: [0.75, 1.0],
      2: [0.75, 0.15],
      3: [0.5, -0.5],
      4: [1.0, -0.5]}

"""Adicionar ligações"""
for i in range(1, n+1):
    for j in range(1, n+1):
        if j!=i:
            G.add_edge(i,j)


"""Posição dos nós"""
pos = None
list1 = [[None]]
for i in range(1, n):

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


for i in range(n):
    if pos == None:
        pos = {
            i+1: list1[i]
            }
    else:
        pos[i+1] = list1[i]

pos = {1: [0.75, 1.0],
      2: [0.75, 0.15],
      3: [0.5, -0.5],
      4: [1.0, -0.5]} 
      
"""Etiquetas do grafo"""

edge_labels = None
labels1 = None
labels = []

for i in range(1, n):
    for j in range(1, n+1): 
        if j > i:
            l1 = '$\\phi_{'
            l2 = '\\to '
            l3 = '}$\n $\\phi_{'
            l4 = '\\to '
            l5 = '}$'
            labels1 = l1+str(j)+l2+str(i)+l3+str(i)+l4+str(j)+l5
            labels.append(labels1)

x = 0
for i in range(1, n):
    for j in range(1, n+1):
        
        if j > i:
            if x == 0:
                edge_labels = {
                (i, j): labels[x]
                }
            else: 
                edge_labels[i, j] = labels[x]
            x += 1


"""Grafo"""
fig = plt.figure(1, figsize=(14, 10)) 
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos, connectionstyle='arc3, rad = 0.2', 
        node_size=3000, arrowsize=14, arrowstyle='simple', font_size=30)

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20, bbox=dict(alpha=0))
 
plt.show()


def unitario_controlado(qc, qubits: list, phases: list): # x,y,z = Qubit; a,b,c,d = Fases
    qc.cp(phases[2]-phases[0], qubits[0], qubits[1]) # controlado-U1(c-a)
    qc.p(phases[0], qubits[0]) # U1(a)
    qc.cp(phases[1]-phases[0], qubits[0], qubits[2]) # controlado-U1(b-a)
    
    # controlado do controlado U1(d-c+a-b)
    qc.cp((phases[3]-phases[2]+phases[0]-phases[1])/2, qubits[1], qubits[2])
    qc.cx(qubits[0], qubits[1])
    qc.cp(-(phases[3]-phases[2]+phases[0]-phases[1])/2, qubits[1], qubits[2])
    qc.cx(qubits[0], qubits[1])
    qc.cp((phases[3]-phases[2]+phases[0]-phases[1])/2, qubits[0], qubits[2])


def U(times, qc, unit, eigen, ases: list): # a,b,c = fases do U1; d,e,f = fases do U2; g,h,i = fases do U3; j,k,l = fases do U4; lista=[m, n, o, p, q, r, s, t, u, a, b, c, d, e, f, g, h, i, j, k, l]
    unitario_controlado(qc, [unit[0]]+eigen[0:2], [0]+phases[0:3])
    unitario_controlado(qc, [unit[0]]+eigen[2:4], [phases[3]]+[0]+phases[4:6])
    unitario_controlado(qc, [unit[0]]+eigen[4:6], phases[6:8]+[0]+[phases[8]])
    unitario_controlado(qc, [unit[0]]+eigen[6:8], phases[9:12]+[0])


def U_final(times, eigen, phases: list):
    unit = QuantumRegister(1, 'unit')
    qc = QuantumCircuit(unit, eigen)
    for _ in range(2**times):
        U(times, qc, unit, eigen, phases)
    return qc.to_gate(label='U'+'_'+(str(2**times)))


# Autovalores
# TODO: gerador de auto valores automatico

auto_valor = ["11000110", "10001101", "11001001"]

# Portas especificas dos autoestados
def autoestado(qc, eigen, index):
    for i in range(0, len(eigen)):
        if auto_valor[index][i] == '1':
            qc.x(eigen[i])
        if auto_valor[index][i] == '0':
            pass
    qc.barrier()
    return qc


# Inicialização
unit = QuantumRegister(6, 'unit')
eigen = QuantumRegister(8, 'eigen')
classico = ClassicalRegister(6, 'classico')
qc = QuantumCircuit(unit, eigen, classico)



# Usando um autoestado específico
autoestado(qc, eigen, 0)



# Hadamard nos qubits controles
qc.h(unit[:])
qc.barrier()


# Unitário controlado
phases = [pi / 2, pi / 8, pi / 4, pi / 2, pi / 4, pi / 4, pi / 8, pi / 4, pi / 8, pi / 4, pi / 4, pi / 8] # a, b, c, d, e, f, g, h, i, j, k, l
for i in range(0, 6):
    qc.append(U_final(i, eigen, phases), [unit[5-i]] + eigen[:])


# Inverse QFT 
qc.barrier()
qft = QFT(num_qubits=len(unit), inverse=True, insert_barriers=True, do_swaps=False, name='Inverse QFT')
qc.append(qft, qc.qubits[:len(unit)])
qc.barrier()


# Medida
qc.measure(unit, classico)


# Circuito
qc.draw()

# Histograma
backend = Aer.get_backend('qasm_simulator')
shots = 10000
t_qc = transpile(qc, backend)
qobj = assemble(t_qc, shots=shots)
resuts = backend.run(qobj).result()
answer = resuts.get_counts()


print(answer)
plot_histogram(answer)

