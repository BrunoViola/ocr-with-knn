from PIL import Image
import numpy as np
import os

from src.variaveis import remover_bordas_flag

def remover_bordas(img):
   img_np = np.array(img)
   #cria uma máscara com os pixels não brancos (fazem parte do caractere)
   mask = img_np < 130

   #encontra os limites da máscara
   coords = np.argwhere(mask) # coordenadas dos pixels não brancos
   y0, x0 = coords.min(axis=0) # encontra o menor valor de y e x
   y1, x1 = coords.max(axis=0) + 1  # maior valor de y e x, soma 1 para incluir a borda final

   # corte da imagem
   recortada = img.crop((x0, y0, x1, y1))
   return recortada

def pre_processar_imagem(img, dimensao_imagem):
   img = img.convert('L')  # converte para escala de cinza
   if remover_bordas_flag:
      img = remover_bordas(img)  # remove bordas brancas
   img = img.resize((dimensao_imagem, dimensao_imagem))  # redimensiona para 32
   return img

#img = Image.open('dataset/0/AMBV_1_59.jpeg')  # exemplo de uso
#img = pre_processar_imagem(img, 32)  # pré-processa a imagem
#img.show()  # exibe a imagem pré-processada