clear all; close all; clc;

% 1. Load and preprocess the image
A = imread("ORIGA/Images_Cropped/001.jpg");
A = imresize(A, [512, 512]); % Toolbox Wavelets requires power-of-2 dimensions (2^9 = 512)
B = double(rgb2gray(A));     % Convert to double precision for wavelet math

% 2. Configure wavelet options ('db3' = Daubechies with 3 vanishing moments)
options.wavelet_type = 'daubechies';
options.wavelet_vm = 3;

% 3. Set decomposition scale
% For a 512x512 image (Jmax = 9), n = 2 decomposition levels means Jmin = 9 - 2 = 7
n = 2; 
Jmax = log2(size(B, 1));
Jmin = Jmax - n;

% 4. Perform 2D forward wavelet transform (+1 = forward, -1 = inverse)
MW = perform_wavelet_transform(B, Jmin, +1, options);

% 5. Plot the square wavelet decomposition
plot_wavelet(MW, Jmin);
colormap(flipud(gray));
axis off;
axis tight;
