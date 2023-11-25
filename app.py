import cv2
import numpy as np
from sklearn.cluster import KMeans

def segmentacao_watershed(imagem):
    imagemmCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imagemmCinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
    cv2.watershed(imagem, markers)
    imagem[markers == -1] = [0, 0, 255]

    return imagem

def segmentacao_kmeans(imagem, k=3):
    imagemmRemodelada = imagem.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imagemmRemodelada)
    imagemSegmentada = kmeans.cluster_centers_[kmeans.labels_]
    imagemSegmentada = imagemSegmentada.reshape(imagem.shape).astype(np.uint8)
    
    return imagemSegmentada

def segmentacao_mean_shift(imagem):
    mean_shift = cv2.pyrMeanShiftFiltering(imagem, 21, 51)
    return mean_shift

# Carregar imagens
imagem1 = cv2.imread('esfregaco_01.jpeg')
imagem2 = cv2.imread('esfregaco_02.jpeg')
imagem3 = cv2.imread('esfregaco_03.jpeg')

# # Segmentação utilizando Watershed
# result_watershed1 = segmentacao_watershed(imagem1.copy())
# result_watershed2 = segmentacao_watershed(imagem2.copy())
# result_watershed3 = segmentacao_watershed(imagem3.copy())

# cv2.imshow('Resultado Whatershed Imagem 1', result_watershed1)
# cv2.imshow('Resultado Whatershed Imagem 2', result_watershed2)
# cv2.imshow('Resultado Whatershed Imagem 3', result_watershed3)

# # Segmentação utilizando K-means
# result_kmeans1 = segmentacao_kmeans(imagem1.copy())
# result_kmeans2 = segmentacao_kmeans(imagem2.copy())
# result_kmeans3 = segmentacao_kmeans(imagem3.copy())

# cv2.imshow('Resultado K-means Imagem 1', result_kmeans1)
# cv2.imshow('Resultado K-means Imagem 2', result_kmeans2)
# cv2.imshow('Resultado K-means Imagem 3', result_kmeans3)

# Segmentação utilizando Mean Shift
result_mean_shift1 = segmentacao_mean_shift(imagem1.copy())
result_mean_shift2 = segmentacao_mean_shift(imagem2.copy())
result_mean_shift3 = segmentacao_mean_shift(imagem3.copy())

cv2.imshow('Resultado Mean Shift Imagem 1', result_mean_shift1)
cv2.imshow('Resultado Mean Shift Imagem 2', result_mean_shift2)
cv2.imshow('Resultado Mean Shift Imagem 3', result_mean_shift3)

cv2.waitKey(0)
cv2.destroyAllWindows()
