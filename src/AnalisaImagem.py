import numpy as np
import pandas as pd
from joblib import load
from keras.preprocessing import image
import cv2 as cv
from Camera import Camera

frame = Camera.frameCapter()

imagem = image.load_img('maoaberta1.jpg', target_size= (28, 28))
imagem = np.array(imagem)
imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
imagemaberta = imagem.reshape((-1,28,28,1))

imagem = image.load_img('maofechada1.jpg', target_size= (28, 28))
imagem = np.array(imagem)
imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
imagemfechada = imagem.reshape((-1,28,28,1))

model = load("machinelearning.joblib")

y_predaberta = model.predict(imagemaberta)
y_predfechada = model.predict(imagemfechada)
print(y_predaberta, y_predfechada)