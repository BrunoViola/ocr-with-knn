import matplotlib.pyplot as plt
import numpy as np

def plotar_metricas_por_classe(relatorio_por_classe):
   classes = []
   precisoes = []
   revocoes = []
   f1s = []

   for classe in sorted(relatorio_por_classe.keys(), key=int):
      classes.append(classe)
      precisoes.append(np.mean(relatorio_por_classe[classe]['precision']))
      revocoes.append(np.mean(relatorio_por_classe[classe]['recall']))
      #f1s.append(np.mean(relatorio_por_classe[classe]['f1-score']))

   x = np.arange(len(classes))  # posições no eixo x
   largura = 0.25

   plt.figure(figsize=(14, 6))
   plt.bar(x - largura, precisoes, width=largura, label='Precisão', color='royalblue')
   plt.bar(x, revocoes, width=largura, label='Revocação', color='orange')
   #plt.bar(x + largura, f1s, width=largura, label='F1-Score', color='green')

   plt.xticks(x, classes)
   plt.xlabel("Classe")
   plt.ylabel("Métrica")
   plt.title("Métricas por Classe - Média de 10 Iterações")
   plt.legend()
   plt.tight_layout()
   plt.show()
