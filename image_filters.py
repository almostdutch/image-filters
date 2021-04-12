'''
image_filters.py

Arsenal of image processing filters:
    
* Histogram equalization

* Local histogram equalization

* Spatially adaptive smoothing filters (preserve edges):
  * flat kernel (application: uniform and Gaussian noise)
  * gaussian kernel (application: uniform and Gaussian noise)
  
* Order statistic filters
  * Median kernel (application: impulsive noise)
  * Min kernel (application: salt noise)
  * Max kernel (application: pepper noise)
  * Mid-point kernel (application: uniform and Gaussian noise)
  * Alpha-trimmed-mean kernel (application: Gaussian + impulsive noise)
  
* Spatially adaptive order statistic filters (preserve edges):
  * Median kernel (application: impulsive noise)
  * Min kernel (application: salt noise)
  * Max kernel (application: pepper noise)
  * Mid-point kernel (application: uniform and Gaussian noise)
  * Alpha-trimmed-mean kernel (application: Gaussian + impulsive noise)
  
* High boost filter (application: edge enhancement and de-enhancement)

* Homomorphic filter (application: contrast enhancement in dark and bright regions)
'''

import numpy as np
from numpy.matlib import repmat
from scipy.signal import convolve2d

def CapIntensity(img_in, bpp):
    img_out = img_in;
    img_out[img_out < 0] = 0;
    img_out[img_out > (bpp - 1)] = bpp - 1;
    return img_out;

def ImageNormalization(img_in, normalization, bpp):
    if normalization == 'bpp':
        N = bpp - 1;
        img_out = (img_in - np.mean(img_in)) / N;    
    if normalization == 'range':
        N = np.max(img_in) - np.min(img_in);
        img_out = (img_in - np.mean(img_in)) / N;
    if normalization == 'std':
        N = np.std(img_in);
        img_out = (img_in - np.mean(img_in)) / N;    
    if normalization == 'sos':
        temp = img_in - np.mean(img_in)
        N = np.sqrt(np.sum(np.power(temp, 2)));
        img_out = temp / N; 
    return img_out, N;

def GaussianKernel1D(kernel_size, sigma):
    # returns 1D Gaussian kernel
    
    kn = int((kernel_size - 1) / 2);
    X = np.arange(-kn, kn + 1, 1);
    kernel = np.exp(-(np.power(X, 2)) / (2 * sigma ** 2));
    kernel = kernel / kernel.sum();
    return kernel;

def FlatKernel1D(kernel_size):
    # returns 1D flat kernel
    
    kernel = np.ones((1, kernel_size)) / kernel_size;
    return kernel;

def GaussianKernel2D(kernel_size, sigma):
    # returns 2D Gaussian kernel
    
    kn = int((kernel_size - 1) / 2);
    x = np.arange(-kn, kn + 1, 1);
    [X, Y] = np.meshgrid(x, x, sparse=False, indexing='xy');
    kernel = np.exp(-(np.power(X, 2) + np.power(Y, 2)) / (2 * sigma ** 2));
    kernel = kernel / kernel.sum();
    return kernel;

def FlatKernel2D(kernel_size):
    # returns 2D flat kernel
    
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2;
    return kernel;

def TakeCareOfBoundaries(img_in, kernel_size, boundary = 'edge', fill_value = None):
    # takes care of image boundaries and returns a properly padded image
    # boundary - boundary condition: edge or fill (with fill_value)
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    kn = int((kernel_size - 1) / 2);
    img_in_padded = np.zeros((nr + 2 * kn, nc + 2 * kn));        
    img_in_padded[kn:kn + nr, kn:kn + nc] = img_in;
    
    if boundary == 'edge':
        # up and down
        img_in_padded[0:kn, kn:-kn] = repmat(img_in[0, :], kn, 1);
        img_in_padded[-kn:, kn:-kn] = repmat(img_in[-1, :], kn, 1);
        
        # left and right
        img_in_padded[kn:- kn, 0:kn] = repmat(img_in[:, 0], kn, 1).T;
        img_in_padded[kn:- kn, -kn:] = repmat(img_in[:, -1], kn, 1).T;
        
        # corners
        img_in_padded[0:kn, 0:kn] = repmat(img_in[0, 0], kn, kn);
        img_in_padded[0:kn, -kn:] = repmat(img_in[0, -1], kn, kn);
        img_in_padded[-kn:, 0:kn] = repmat(img_in[-1, 0], kn, kn);
        img_in_padded[-kn:, -kn:] = repmat(img_in[-1, -1], kn, kn);
        
    if boundary == 'fill':
        if fill_value == None:
            fill_value = 0;
        # up and down
        img_in_padded[0:kn, kn:-kn] = fill_value;
        img_in_padded[-kn:, kn:-kn] = fill_value;
        
        # left and right
        img_in_padded[kn:- kn, 0:kn] = fill_value;
        img_in_padded[kn:- kn, -kn:] = fill_value;
        
        # corners
        img_in_padded[0:kn, 0:kn] = fill_value;
        img_in_padded[0:kn, -kn:] = fill_value;
        img_in_padded[-kn:, 0:kn] = fill_value;
        img_in_padded[-kn:, -kn:] = fill_value;      
        
    return img_in_padded;

def HistEqualization(img_in, bpp):
    # performs global histogram equalization
    # bpp - bits per pixel
    
    histogram, _ = np.histogram(img_in, bins = bpp, range = (0, bpp-1));

    pmf = histogram / histogram.sum(); # probability mass function
    cdf = np.cumsum(pmf); # cumulative distribution function
    Tp = cdf; # point-wise intensity transformation operator
    
    img_out = np.zeros(img_in.shape);
    for row_no in range(img_in.shape[0]):
        for col_no in range(img_in.shape[1]):
            img_out[row_no, col_no] = Tp[int(np.round(img_in[row_no, col_no]))] * (bpp - 1);
            
    return img_out;

def LocalHistEqualization(img_in, bpp, kernel_size, boundary = 'edge', fill_value = None):
    # performs local neighborhood histogram equalization
    # bpp - bits per pixel
    # kernel_size - kernel size (3, 5, ..)
    # boundary - boundary condition: edge or fill (with fill_value)
    
    img_out = np.zeros(img_in.shape);  
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    kn = int((kernel_size - 1) / 2);
    img_in_padded = TakeCareOfBoundaries(img_in, kernel_size, boundary, fill_value);
    
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii:ii + kernel_size, jj:jj + kernel_size];
            nb_processed = HistEqualization(nb, bpp);
            img_out[ii, jj] = nb_processed[kn, kn];
            
    return img_out;

def SpatiallyAdaptiveSmoothingFilter(img_in, noise_sigma, kernel_size, filter_type, kernel_sigma = None, boundary = 'edge', fill_value = None):
    # performs spatially adaptive filtering (preserves edges)
    # noise_sigma - noise standard deviation estimated from a flat image region
    # kernel_size - kernel size
    # filter_type - filter type: flat or gaussian (with kernel_sigma)
    # kernel_sigma - sigma for gaussian kernel
    # boundary - boundary condition: edge or fill (with fill_value)
    # application: uniform and Gaussian noise
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    kn = int((kernel_size - 1) / 2);

    if filter_type == 'flat':
        kernel = FlatKernel1D(kernel_size);
    if filter_type == 'gaussian':
        kernel = GaussianKernel1D(kernel_size, kernel_sigma);
    
    # 1D filtering in X dir
    img_in_padded = TakeCareOfBoundaries(img_in, kernel_size, boundary, fill_value);  
    img_out = np.zeros(img_in.shape);
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii + kn, jj:jj + kernel_size];
            nb_sigma = np.std(nb);
            nb_filtered = np.sum(nb * kernel);
            img_out[ii, jj] = (1 - (noise_sigma / nb_sigma) ** 2) * nb[kn] + (noise_sigma / nb_sigma) ** 2 * nb_filtered;

    # 1D filtering in Y dir
    img_in_padded = TakeCareOfBoundaries(img_out, kernel_size, boundary, fill_value); 
    img_out = np.zeros(img_in.shape);
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii:ii + kernel_size, jj + kn];
            nb_sigma = np.std(nb);
            nb_filtered = np.sum(nb * kernel);           
            img_out[ii, jj] = (1 - (noise_sigma / nb_sigma) ** 2) * nb[kn] + (noise_sigma / nb_sigma) ** 2 * nb_filtered;

    return img_out;

def OrderStatisticFilter(img_in, kernel_size, filter_type, alpha = None, boundary = 'edge', fill_value = None):
    # performs nonlinear filtering
    # kernel_size - kernel size (3, 5, ..)
    # filter_type - filter type: median, min, max, mid-point, or alpha-trimmed-mean (with alpha)
    # alpha - delete N = alpha (1, 2, ..) numbers at the begining and at the end of the array 
    # boundary - boundary condition: edge or fill (with fill_value)
    # application: median (impulsive noise), min (salt noise), max (pepper noise)
    #   mid-point (uniform and Gaussian noise), alpha-trimmed-mean (Gaussian + impulsive noise)
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    kn = int((kernel_size - 1) / 2);
    
    # 1D filtering in X dir
    img_in_padded = TakeCareOfBoundaries(img_in, kernel_size, boundary, fill_value);  
    img_out = np.zeros(img_in.shape);
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii + kn, jj:jj + kernel_size];
            
            if filter_type == 'median':
                nb_filtered = np.median(nb);
            if filter_type == 'min':
                nb_filtered = np.min(nb);
            if filter_type == 'max':
                nb_filtered = np.max(nb);                
            if filter_type == 'mid-point':
                nb_filtered = (np.min(nb) + np.max(nb)) / 2;  
            if filter_type == 'alpha-trimmed-mean':
                nb_sorted = np.sort(nb);
                nb_filtered = np.sum(nb_sorted[int(alpha):int(kernel_size - alpha)]) / (kernel_size - 2 * alpha);
                
            img_out[ii, jj] = nb_filtered;   

    # 1D filtering in Y dir
    img_in_padded = TakeCareOfBoundaries(img_out, kernel_size, boundary, fill_value); 
    img_out = np.zeros(img_in.shape);
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii:ii + kernel_size, jj + kn];
            
            if filter_type == 'median':
                nb_filtered = np.median(nb);
            if filter_type == 'min':
                nb_filtered = np.min(nb);
            if filter_type == 'max':
                nb_filtered = np.max(nb);                
            if filter_type == 'mid-point':
                nb_filtered = (np.min(nb) + np.max(nb)) / 2;  
            if filter_type == 'alpha-trimmed-mean':
                nb_sorted = np.sort(nb);
                nb_filtered = np.sum(nb_sorted[int(alpha):int(kernel_size - alpha)]) / (kernel_size - 2 * alpha);  
                
            img_out[ii, jj] = nb_filtered;             

    return img_out;

def SpatiallyAdaptiveOrderStatisticFilter(img_in, noise_sigma, kernel_size, filter_type, alpha = None, boundary = 'edge', fill_value = None):
    # performs spatially adaptive nonlinear filtering (preserves edges)
    # noise_sigma - noise standard deviation estimated from a flat image region
    # kernel_size - kernel size (3, 5, ..)
    # filter_type - filter type: median, min, max, mid-point, or alpha-trimmed-mean (with alpha)
    # alpha - delete N = alpha (1, 2, ..) numbers at the begining and at the end of the array 
    # boundary - boundary condition: edge or fill (with fill_value)
    # application: median (impulsive noise), min (salt noise), max (pepper noise)
    #   mid-point (uniform and Gaussian noise), alpha-trimmed-mean (Gaussian + impulsive noise)
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    kn = int((kernel_size - 1) / 2);

    # 1D filtering in X dir
    img_in_padded = TakeCareOfBoundaries(img_in, kernel_size, boundary, fill_value);  
    img_out = np.zeros(img_in.shape);
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii + kn, jj:jj + kernel_size];
            nb_sigma = np.std(nb);
            
            if filter_type == 'median':
                nb_filtered = np.median(nb);
            if filter_type == 'min':
                nb_filtered = np.min(nb);
            if filter_type == 'max':
                nb_filtered = np.max(nb);                
            if filter_type == 'mid-point':
                nb_filtered = (np.min(nb) + np.max(nb)) / 2;  
            if filter_type == 'alpha-trimmed-mean':
                nb_sorted = np.sort(nb);
                nb_filtered = np.sum(nb_sorted[int(alpha):int(kernel_size - alpha)]) / (kernel_size - 2 * alpha);
                
            img_out[ii, jj] = (1 - (noise_sigma / nb_sigma) ** 2) * nb[kn] + (noise_sigma / nb_sigma) ** 2 * nb_filtered;

    # 1D filtering in Y dir
    img_in_padded = TakeCareOfBoundaries(img_out, kernel_size, boundary, fill_value); 
    img_out = np.zeros(img_in.shape);
    for ii in range(0, img_out.shape[0]):
        for jj in range(0, img_out.shape[1]):
            nb = img_in_padded[ii:ii + kernel_size, jj + kn];
            nb_sigma = np.std(nb);
            
            if filter_type == 'median':
                nb_filtered = np.median(nb);
            if filter_type == 'min':
                nb_filtered = np.min(nb);
            if filter_type == 'max':
                nb_filtered = np.max(nb);                
            if filter_type == 'mid-point':
                nb_filtered = (np.min(nb) + np.max(nb)) / 2;  
            if filter_type == 'alpha-trimmed-mean':
                nb_sorted = np.sort(nb);
                nb_filtered = np.sum(nb_sorted[int(alpha):int(kernel_size - alpha)]) / (kernel_size - 2 * alpha);
                
            img_out[ii, jj] = (1 - (noise_sigma / nb_sigma) ** 2) * nb[kn] + (noise_sigma / nb_sigma) ** 2 * nb_filtered;      

    return img_out;

def HighBoostFilter(img_in, alpha, method, bpp):
    # performs high boost filtering (edge enhancement)
    # method - method for edge enhancement: laplacian or gaussian
    # alpha - edge enhancement factor (alpha > 0 for enhancement and alpha < 0 for de-enhancement)
    # bpp - bits per pixel
    
    if method == 'laplacian':
        kernel = np.array([[-alpha, -alpha, -alpha], 
                           [-alpha, alpha * 8 + 1, -alpha],
                           [-alpha, -alpha, -alpha]]);
        
        img_out = convolve2d(img_in, kernel, boundary = 'symm', mode = 'same');
        img_out = CapIntensity(img_out, bpp);
        
    if method == 'gaussian':
        kernel = GaussianKernel2D(kernel_size = 7, sigma = 3);
        img_in_LP = convolve2d(img_in, kernel, boundary = 'symm', mode = 'same');
        img_out = img_in + alpha * (img_in - img_in_LP);
        img_out = CapIntensity(img_out, bpp);
    
    return img_out;

def HomomorphicFilter(img_in, alpha1, alpha2, bpp):
    # performs homomorphic filtering (contrast enhancement in black and bright regions)
    # alpha1 - weighting factor for illumination image (alpha1 <= 1)
    # alpha2 - weighting factor for reflectance image (alpha2 > 1)
    # bpp - bits per pixel
    
    kernel_size = 7;
    kn = int((kernel_size - 1) / 2);
    sigma = 3;
    kernel_AP = np.zeros((kernel_size, kernel_size));
    kernel_AP[kn, kn] = 1;
    kernel_LP = GaussianKernel2D(kernel_size, sigma);
    kernel_HP = kernel_AP - kernel_LP;
    
    log_img_in = np.log(img_in);
    log_img_illumination = convolve2d(log_img_in, kernel_LP, boundary = 'symm', mode = 'same');
    log_img_reflectance = convolve2d(log_img_in, kernel_HP, boundary = 'symm', mode = 'same');
    
    img_illumination = np.exp(alpha1 * log_img_illumination);
    img_reflectance = np.exp(alpha2 * log_img_reflectance);
    
    img_out = img_illumination * img_reflectance;
    img_out = CapIntensity(img_out, bpp);
    
    return img_out;
    
    
    
    
    
    
    
    
    
    