function subband = WaveletDecomp(image)
 % Dekomposisi DWT 2D, wavelet Haar, 2 level
    [c, s] = wavedec2(double(image), 2, 'haar');
    
    % Mengambil 7 subband DWT
    LL2 = appcoef2(c, s, 'haar', 2);
    [H2, V2, D2] = detcoef2('all', c, s, 2);   % level 2
    [H1, V1, D1] = detcoef2('all', c, s, 1);   % level 1
    
    subband = {
        LL2;   % LL2
        H2;    % LH2
        V2;    % HL2
        D2;    % HH2
        H1;    % LH1
        V1;    % HL1
        D1     % HH1
    };
end

