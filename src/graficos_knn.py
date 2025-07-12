import matplotlib.pyplot as plt
import numpy as np
from src.variaveis import dimensao_imagem, data_augmentation_flag, remover_bordas_flag

# manipulação do nome do arquivo para salvar o gráfico
modo = 'COM_DataAug' if data_augmentation_flag else 'SEM_DataAug'
remocao_bordas = 'Bordas_REMOVIDAS' if remover_bordas_flag else 'Bordas_NAO_REMOVIDAS'


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
   plt.savefig(f"metricas_por_classe/mpc_{dimensao_imagem}x{dimensao_imagem}_{modo}_{remocao_bordas}.png")
   plt.show()
