from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt


lena = load_image('lena.jpg')
clean_image = median(lena, ball(3))
edgeMAG = edge_detection(clean_image)

#plt.hist(edgeMAG.flatten(), bins = 256, range =(0,255));

edge_binary = edgeMAG > 40
plt.imshow(edge_binary, cmap = 'gray')


edge_image = Image.fromarray(edge_binary)
edge_image.save('my_edges.png')
