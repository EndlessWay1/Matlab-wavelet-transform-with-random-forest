function imageAug = AugmentasiCitra(image)
    % Fungsi untuk melakukan augmentasi citra (grayscale)

    % Salin citra asli
    imageAug = image;

    % Horizontal flip secara acak
    if rand < 0.5
        imageAug = fliplr(imageAug);
    end

    % Rotasi dengan sudut acak antara -15 sampai 15 derajat
    sudut = -15 + 30 * rand;

    [tinggi, lebar] = size(imageAug);
    cx = floor(lebar / 2) + 1;    % titik tengah (indeks 1-based)
    cy = floor(tinggi / 2) + 1;

    theta = deg2rad(sudut);
    a = cos(theta);
    b = sin(theta);

    % Grid koordinat output, lalu petakan balik ke koordinat sumber
    % (arah rotasi sama dengan cv2.getRotationMatrix2D: sudut positif = CCW)
    [xo, yo] = meshgrid(1:lebar, 1:tinggi);
    xs =  a * (xo - cx) - b * (yo - cy) + cx;
    ys =  b * (xo - cx) + a * (yo - cy) + cy;

    % Pantulkan koordinat di luar citra (setara BORDER_REFLECT_101)
    xs = reflect101(xs, lebar);
    ys = reflect101(ys, tinggi);

    % Interpolasi bilinear -> hasil halus, tanpa padding nol
    out = interp2(1:lebar, 1:tinggi, double(imageAug), xs, ys, 'linear');

    % Kembalikan ke tipe data semula (mis. uint8)
    imageAug = cast(out, 'like', image);
end

function p = reflect101(p, n)
    % Pantulan kontinu pada rentang [1, n] tanpa mengulang piksel tepi
    if n == 1
        p = ones(size(p));
        return;
    end
    periode = 2 * (n - 1);
    p = mod(p - 1, periode);
    idx = p > (n - 1);
    p(idx) = periode - p(idx);
    p = p + 1;
    p = min(max(p, 1), n);   % pengaman numerik
end