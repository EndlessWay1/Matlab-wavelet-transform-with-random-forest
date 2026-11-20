function [HL1, LH1, HH1, LL2, HL2, LH2, HH2] = WaveletConvertLvl2(data)

    % Configure wavelet
    options.wavelet_type = 'daubechies';
    options.wavelet_vm = 3;

    % Convert to double
    data = double(data);

    % Determine decomposition level
    Jmax = log2(size(data, 1));
    n = 2;
    Jmin = Jmax - n;

    % Forward wavelet transform
    MW = perform_wavelet_transform(data, Jmin, +1, options);

    % Extract individual subbands
    bands = convert_wavelets2list(MW, Jmin);
    HL1 = bands{1};
    LH1 = bands{2};
    HH1 = bands{3};

    LL2 = bands{4};
    HL2 = bands{5};
    LH2 = bands{6};
    HH2 = bands{7};
end