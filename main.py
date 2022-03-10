#!  /usr/bin/python3

import sys
import re
import numpy as np
from math import sqrt
from skimage.feature import blob_log
from skimage import img_as_uint
from skimage.measure import regionprops
from skimage.restoration import rolling_ball
from skimage.io import imread, imshow
from skimage.filters import try_all_threshold, gaussian, threshold_otsu, threshold_local
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

filename_dapi = sys.argv[1]
filename_mcherry = re.sub("_DAPI", "_mCherry", filename_dapi)

print("Processing file: ", filename_dapi)

img_dapi = imread(filename_dapi, as_gray=True)
img_dapi_thresheld = img_dapi > threshold_local(img_dapi, block_size=201, param=50)

distance = ndi.distance_transform_edt(img_dapi_thresheld)
coords = peak_local_max(distance, min_distance=25, footprint=np.ones((25, 25)), labels=img_dapi_thresheld)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=img_dapi_thresheld)

print("Loading mCherry file:", filename_mcherry)
img_mcherry = imread(filename_mcherry, as_gray=True)
labels_picked = labels.copy()
regions = regionprops(labels_picked)
labels_picked_shape = labels_picked.shape
f = open(filename_mcherry+"_inten.tsv", "w")
counter = 1
for region in regions:
    region_centroid = region.centroid # row, col
    if region.area < 2000 or region.convex_area > 8500 or \
       region.area / region.bbox_area < 0.5 or \
       region.eccentricity > 0.8 or \
       region_centroid[0] / labels_picked_shape[0] < 0.05 or \
       region_centroid[0] / labels_picked_shape[0] > 0.95 or \
       region_centroid[1] / labels_picked_shape[1] < 0.05 or \
       region_centroid[1] / labels_picked_shape[1] > 0.95:
        labels_picked[region.slice] = labels_picked[region.slice] * np.invert(region.image)
    else:
        region_sum = int(np.round(np.sum(img_mcherry[region.slice] * region.image)))
        sliced_image = img_mcherry[region.slice]
        blobs_log = blob_log(sliced_image, min_sigma=1, max_sigma=3, threshold=0.04, overlap=0.5)
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)
        ax[0].imshow(sliced_image)
        ax[0].set_title("intensity_image")
        ax[0].axis("off")
        ax[1].imshow(sliced_image)
        ax[1].set_title("blobs")
        ax[1].axis("off")
        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), r, color="yellow", linewidth=2, fill=False)
            ax[1].add_patch(c)
        num_blobs = len(blobs_log)
        print("region.convex_area, region_sum, num_blobs: ", region.convex_area, region_sum, num_blobs, sep="\t")
        plt.savefig(filename_mcherry+"_blob_"+str(counter)+"_"+str(num_blobs)+".png", dpi=300)
        plt.close()
        counter +=1
        f.write(filename_mcherry + "\t" + str(region_sum) + "\t" + str(num_blobs) + "\n")

f.close()

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)
ax[0].imshow(labels)
ax[0].set_title("labels")
ax[0].axis('off')
ax[1].imshow(labels_picked)
ax[1].set_title("labels_picked")
ax[1].axis('off')
plt.savefig(filename_dapi+"_segmented.png", dpi=300)
plt.close()


