# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

from torchvision import datasets
import numpy as np

train_data = datasets.CIFAR10(root = "/home/bhavika/Users/Bhavika/BITS - Acads/Computer_Vision/100-Days-of-ML/data", train = True, download = True)
img, label = train_data[0]
img_array = np.array(img)

#red = img_array[:, :, 0].astype(float) # just the red channel

#horizontal edge detector:
# note here: this filt has shape as (channels, height, width)
# filt = np.array([[[-1, -1, -1],
#                  [0, 0, 0],
#                  [1, 1, 1]],
#                  [[-1, -1, -1],
#                  [0, 0, 0],
#                  [1, 1, 1]],
#                  [[-1, -1, -1],
#                  [0, 0, 0],
#                  [1, 1, 1]]])

#we want filt to have ( height, width, channels) because that matches the image shape; so we write it like that here
filt_1 = np.array([
    [[-1,-1,-1], [-1,-1,-1], [-1,-1,-1]],
    [[ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0]],
    [[ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1]]
])

#vertical edge detector:
filt_2 = np.array([
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
])

filters = [filt_1, filt_2]

H, W, C= img_array.shape
num_filters = len(filters) # always take the height of the filter; filter's shape is (channels, height, width) = (3, 3, 3)
F = filters[0].shape[0]
S = 1
P = 0
padded = np.pad(img_array, ((P,P), (P,P), (0,0)), mode='constant')
"""
(P, P) for height — add P rows of padding before (on top) and P rows after (on bottom).
(P, P) for width — add P columns before (left) and P columns after (right).
(0, 0) for channels — add zero padding before and after on the channel axis. You do not want to pad channels — that would literally add extra fake color channels, which makes no sense. You only ever pad the two spatial dimensions.
"""
out_size = ((H - F + 2*P)//S) + 1 #// is integer floor division

output = np.zeros((out_size, out_size, num_filters))

for f in range(num_filters):
    curr_filt = filters[f]
    for i in range(out_size):
        for j in range(out_size):
            row_start = i * S
            col_start = j * S
            img_patch = padded[row_start:row_start+F, col_start:col_start+F, :]
            output[i][j][f] = np.sum(curr_filt * img_patch)
            
print(output.shape)   # should be (30, 30,2) because we have 2 filters
#note: elementwise mult op = * or np.multiply(A,B)
#matrix mult op = @ or np.matmul(A,B) or np.dot(A,B)
#print(img_array.shape)
#print(label)
import matplotlib.pyplot as plt
plt.imshow(output[:, :, 0], cmap='gray')
plt.title("Horizontal edge response")
plt.show()

plt.imshow(output[:, :, 1], cmap='gray')
plt.title("Vertical edge response")
plt.show()

#max-pooling:
pool_size = 2
pool_stride = 2

#feature_map = output[:,:, 0] #taking output post filter 1

pooled_maps = []
for k in range(num_filters):
    feature_map = output[:,:,k]
    H_f, W_f = feature_map.shape
    pooled_size = (H_f - pool_size)//pool_stride + 1
    pooled = np.zeros((pooled_size, pooled_size))
    for i in range(pooled_size):
        for j in range(pooled_size):
            row_start = i * pool_stride
            col_start = j * pool_stride
            # extract the 2x2 patch, take its max, store it
            patch = feature_map[row_start:row_start + pool_size, col_start:col_start+pool_size]
            pooled[i][j] = np.max(patch)
            pooled_maps.append(pooled)

plt.imshow(pooled_maps[0], cmap='gray')
plt.title("Pooled - horizontal filter")
plt.show()

plt.imshow(pooled_maps[1], cmap='gray')
plt.title("Pooled - vertical filter")
plt.show()
"""
plt.imshow(output, cmap='gray')
plt.title("Horizontal edge response")
plt.show()"""
"""
plt.imshow(img_array)
plt.title(f"label: {label}")
plt.show()"""

#final size after conv operation would be 30 x 30; if we are using 'n' no.of filters, then it would be 30 x 30 x n

