# -*- coding: utf-8 -*-
"""
Script de pré-processamento modernizado para o ISIC 2018.
Usa a biblioteca Pillow para um processamento de imagem seguro e correto.
"""
import numpy as np
import glob
from PIL import Image
import os

# Parâmetros
height = 256
width  = 256

############################################################# Preparar dados do ISIC 2018 #################################################
# Garanta que este é o caminho correto para a pasta do seu dataset
# Defina a pasta principal do dataset aqui (sem a barra no final)
Dataset_add = 'dataset_isic18' 

# Defina a subpasta das imagens de treino aqui
Tr_add = 'ISIC2018_Task1-2_Training_Input'

# Agora, junte os caminhos de forma segura com os.path.join
# Isso cria o caminho 'dataset_isic18/ISIC2018_Task1-2_Training_Input/*.jpg'
caminho_para_procurar = os.path.join(Dataset_add, Tr_add, '*.jpg')

# A verificação que adicionamos antes
print(f"Procurando por imagens em: {caminho_para_procurar}")
Tr_list = sorted(glob.glob(caminho_para_procurar))
print(f"Número de imagens .jpg encontradas: {len(Tr_list)}")

if len(Tr_list) == 0:
    print("\nERRO CRÍTICO: Nenhuma imagem foi encontrada! Verifique os nomes das pastas.")
    import sys
    sys.exit("Script interrompido.")

num_samples = len(Tr_list)

# Inicializa os arrays que vão guardar os dados
# Usamos uint8 para economizar espaço, pois as imagens são 0-255
Data_train_2018    = np.zeros([num_samples, height, width, 3], dtype=np.uint8)
# Para as máscaras, usaremos 0 e 1, então uint8 também é perfeito
Label_train_2018   = np.zeros([num_samples, height, width], dtype=np.uint8)

print(f'Lendo {num_samples} amostras do ISIC 2018...')

for idx, img_path in enumerate(Tr_list):
    if (idx + 1) % 200 == 0:
        print(f'Processando imagem {idx+1}/{num_samples}')

    # --- Processando a Imagem de Entrada ---
    img = Image.open(img_path)
    # Redimensiona a imagem com interpolação BILINEAR (bom para imagens)
    img_resized = img.resize((width, height), Image.BILINEAR)
    Data_train_2018[idx] = np.array(img_resized)

    # --- Processando a Máscara (Ground Truth) ---
    # Constrói o caminho para o arquivo da máscara correspondente
    base_filename = os.path.basename(img_path).replace('.jpg', '')
    mask_path = os.path.join(Dataset_add, 'ISIC2018_Task1_Training_GroundTruth', f'{base_filename}_segmentation.png')
    
    mask = Image.open(mask_path).convert("L") # Converte para Grayscale (tons de cinza)
    # Redimensiona a máscara com interpolação NEAREST (preserva os valores 0 e 255)
    mask_resized = mask.resize((width, height), Image.NEAREST)
    
    mask_array = np.array(mask_resized)
    
    # Binariza a máscara: tudo que for > 128 (branco) vira 1, o resto fica 0.
    binary_mask = (mask_array > 128).astype(np.uint8)
    Label_train_2018[idx] = binary_mask

print('Leitura do ISIC 2018 finalizada.')

################################################################ Criar os conjuntos de treino/teste ########################################
# Divisão: 1815 para treino, 259 para validação, 520 para teste
print('Dividindo os dados em conjuntos de treino, validação e teste...')

Train_img      = Data_train_2018[0:1815]
Validation_img = Data_train_2018[1815:1815+259]
Test_img       = Data_train_2018[1815+259:num_samples]

Train_mask      = Label_train_2018[0:1815]
Validation_mask = Label_train_2018[1815:1815+259]
Test_mask       = Label_train_2018[1815+259:num_samples]

# Define o caminho de saída
output_path = 'dataset_isic18/'
os.makedirs(output_path, exist_ok=True) # Cria a pasta se não existir
print(f"Salvando arquivos .npy em: {output_path}")

# Salva os arrays de imagens
np.save(os.path.join(output_path, 'data_train.npy'), Train_img)
np.save(os.path.join(output_path, 'data_test.npy'), Test_img)
np.save(os.path.join(output_path, 'data_val.npy'), Validation_img)

# Salva os arrays de máscaras
np.save(os.path.join(output_path, 'mask_train.npy'), Train_mask)
np.save(os.path.join(output_path, 'mask_test.npy'), Test_mask)
np.save(os.path.join(output_path, 'mask_val.npy'), Validation_mask)

print("Pré-processamento completo!")
# Verificação final
print("\nVerificação rápida do arquivo de máscara de treino salvo:")
check_mask = np.load(os.path.join(output_path, 'mask_train.npy'))
print(f"Valores únicos no mask_train.npy: {np.unique(check_mask)}")