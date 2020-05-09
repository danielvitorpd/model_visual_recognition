import numpy as np
from keras.preprocessing import image
import cv2 as cv
from PIL import Image
import pandas as pd
import csv

a2 = []
a1 = []
linhas = []

def writeCSV2(pasta, quantidadeImagens, predicao):
    for i in range(1, quantidadeImagens+1):
        a2 = []
        a2.append(str(predicao))
        img = image.load_img(f'{pasta} ({i}).jpg', target_size= (128, 128))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.ravel()
        img = img.tolist()
        for i in img:
            a2.append(f'{i}')
        linhas.append(a2)

def writeCSV(pasta, quantidadeImagens, predicao):
    for i in range(1, quantidadeImagens+1):
        a2 = []
        a2.append(str(predicao))
        img = image.load_img(f'{pasta} ({i}).png', target_size= (128, 128))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.ravel()
        img = img.tolist()
        for i in img:
            a2.append(f'{i}')
        linhas.append(a2)
        

a1.append("label")
for i in range(1, 16385):
    a1.append(f'pixel{i}')
with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, delimiter=',')
    writer.writerow(a1)
linhas.append(a1)


writeCSV2('maoAberta/maoaberta',39, 0)
writeCSV2('maoFechada/maofechada', 47, 1)
with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, delimiter=',')
    for row in linhas:
        writer.writerow(row)

conteudo = pd.read_csv('test.csv')
print(conteudo)
