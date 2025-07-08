from PIL import Image
import numpy as np
import os

def remover_bordas(img_pil):
   img_np = np.array(img_pil)
   #print(img_np)
    # Cria uma máscara com os pixels não brancos (branco = 255)
   mask = img_np < 130

    # Encontra os limites (bounding box) do conteúdo real
   coords = np.argwhere(mask)
   y0, x0 = coords.min(axis=0)
   y1, x1 = coords.max(axis=0) + 1  # soma 1 para incluir a borda final

    # Corta a imagem
   recortada = img_pil.crop((x0, y0, x1, y1))
   #print(np.array(recortada))
   return recortada

def pre_processar_imagem(img, dimensao_imagem):
   img = img.convert('L')  # converte para escala de cinza
   img = remover_bordas(img)  # remove bordas brancas
   img = img.resize((dimensao_imagem, dimensao_imagem))  # redimensiona para 32
   return img