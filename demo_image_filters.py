'''
demo_image_filters.py
Demo for testing of a multitude of image processing filters in the image domain

'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
from image_filters import CalculateISNR, HistEqualization, LocalHistEqualization, \
    SpatiallyAdaptiveSmoothingFilter, OrderStatisticFilter, SpatiallyAdaptiveOrderStatisticFilter, \
    SpatiallyAdaptiveSmoothingWienerFilter, SpatiallyAdaptiveOrderStatisticWienerFilter, \
    HighBoostFilter, HomomorphicFilter

mu, sigma = 0, 10; # mean and standard deviation of added Gaussian noise
bpp = 2 ** 8; # bits per pixel

# load test image
image_full = np.array(mpimg.imread('test_image.tif'));
image_full = image_full.astype(np.float32);
image_cut = image_full;
# image_cut = image_full[19:99, 79:159];
N1, N2 = image_cut.shape;

# create a linear degradation model
kw = 5;
kernel_degradation = np.zeros((kw, kw));
ind = int((kw - 1) / 2);
kernel_degradation[ind, ind] = 1;

# add independent Gaussian noise
noise = np.random.normal(mu, sigma, size = (N1, N2));
image_cut_add_noise = image_cut + noise; 

# filtering
method = 'SpatiallyAdaptiveSmoothingFilter';
img_in = image_cut_add_noise;
noise_sigma = sigma;
kernel_size = 7;
filter_type = 'gaussian';
kernel_sigma = 3;
boundary = 'edge';
fill_value = 0;
image_restored_1 = SpatiallyAdaptiveSmoothingFilter(img_in, noise_sigma, kernel_size, filter_type, kernel_sigma, boundary, fill_value);
ISNR_1 = CalculateISNR(image_cut, image_cut_add_noise, image_restored_1);
print(method)
print(str(ISNR_1) + '\n')

method = 'SpatiallyAdaptiveOrderStatisticFilter';
img_in = image_cut_add_noise;
noise_sigma = sigma;
kernel_size = 5;
filter_type = 'alpha-trimmed-mean';
alpha = 1;
boundary = 'edge';
fill_value = 0;
image_restored_2 = SpatiallyAdaptiveOrderStatisticFilter(img_in, noise_sigma, kernel_size, filter_type, alpha, boundary, fill_value);
ISNR_2 = CalculateISNR(image_cut, image_cut_add_noise, image_restored_2);
print(method)
print(str(ISNR_2) + '\n')

method = 'SpatiallyAdaptiveSmoothingFilter';
img_in = image_cut_add_noise;
noise_sigma = sigma;
kernel_size = 7;
filter_type = 'gaussian';
kernel_sigma = 3;
boundary = 'edge';
fill_value = 0;
image_restored_3 = SpatiallyAdaptiveSmoothingWienerFilter(img_in, noise_sigma, kernel_size, filter_type, kernel_sigma, boundary, fill_value);
ISNR_3 = CalculateISNR(image_cut, image_cut_add_noise, image_restored_3);
print(method)
print(str(ISNR_3) + '\n')

method = 'SpatiallyAdaptiveOrderStatisticWienerFilter';
img_in = image_cut_add_noise;
noise_sigma = sigma;
kernel_size = 5;
filter_type = 'alpha-trimmed-mean';
alpha = 1;
boundary = 'edge';
fill_value = 0;
image_restored_4 = SpatiallyAdaptiveOrderStatisticWienerFilter(img_in, noise_sigma, kernel_size, filter_type, alpha, boundary, fill_value);
ISNR_4 = CalculateISNR(image_cut, image_cut_add_noise, image_restored_4);
print(method)
print(str(ISNR_4) + '\n')

# show all images
image_restored = image_restored_3;
ISNR = ISNR_3;
fig_width, fig_height = 5, 5;
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(image_cut, cmap='gray')
ax1.set_title("image original")
ax1.set_axis_off()

ax2.imshow(image_cut_add_noise, cmap='gray')
ax2.set_title("image with noise")
ax2.set_axis_off()

ax3.imshow(image_restored, cmap='gray')
ax3.set_title('image restored ISNR = {}'.format(round(ISNR,2)))
ax3.set_axis_off()

ax4.imshow(image_restored - image_cut, cmap='gray')
ax4.set_title("image difference")
ax4.set_axis_off()
plt.tight_layout()
