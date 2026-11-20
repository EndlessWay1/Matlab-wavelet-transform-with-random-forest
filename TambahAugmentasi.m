function [gambar_aug, label_aug] = TambahAugmentasi(gambar_train, label_train, jumlah)

    if nargin < 3
        jumlah = 100;
    end

    gambar_aug = cell(jumlah*2, 1);
    label_aug = zeros(jumlah*2, 1);

    counter = 1;

    % Melakukan augmentasi untuk setiap label
    for label = [0, 1]

        % Mengambil indeks citra berdasarkan label
        indeks_label = find(label_train == label);
        fprintf("Label %d: %d available images\n", ...
            label, length(indeks_label));
        % Memeriksa jumlah citra yang tersedia
        if length(indeks_label) < jumlah
            error('Jumlah citra label %d kurang dari %d', label, jumlah);
        end

        % Memilih citra asli yang berbeda secara acak
        indeks_acak = indeks_label(randperm(length(indeks_label), jumlah));

        
        % Melakukan augmentasi pada setiap citra terpilih
        for k = 1:jumlah
            index = indeks_acak(k);
            
            image = gambar_train{index};
            
            % Melakukan augmentasi
            image_hasil = AugmentasiCitra(image);
            gambar_aug{counter} = image_hasil;
            label_aug(counter) = label;
            counter = counter + 1;

        end
    end

    % Mengacak urutan citra dan label secara bersamaan
    randomizer = randperm(length(label_aug));

    gambar_aug = gambar_aug(randomizer);
    label_aug = label_aug(randomizer);

    % Mengubah label menjadi column vector
    label_aug = label_aug(:);

end