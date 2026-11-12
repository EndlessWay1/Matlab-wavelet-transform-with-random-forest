image = imread("ORIGA/Images_Cropped/001.jpg");
image = imresize(image, [448, 448]);
image = im2double(im2gray(image));        
lambda = 12;
sigma  = 0.56*lambda;
gamma  = 0.5;
psi    = 0;
kernel = 3;

thetas = 0:pi/8:7*pi/8;                 
nT = numel(thetas);
responses = zeros([size(image), nT]);   % 448 x 448 x 8

for k = 1:nT
    gb = GaborFilter(sigma, thetas(k), lambda, psi, gamma, kernel);
    gb = gb - mean(gb(:));
    gb = gb / (norm(gb(:), 2) + eps);
    responses(:,:,k) = conv2(image, gb, 'same');
end

figure;
for k = 1:nT
    subplot(2, 4, k);
    imshow(responses(:,:,k), []);
    title(sprintf("\\theta = %d°", round(rad2deg(thetas(k)))));
end

[maxResp, bestIdx] = max(abs(responses), [], 3);   % strongest response per pixel
figure; imshow(maxResp, []); title("Max response");