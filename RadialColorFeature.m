function f = RadialColorFeature(img)
    img = im2double(img);
    [H, W, ~] = size(img);
    [X, Y] = meshgrid(1:W, 1:H);
    R  = hypot(X - W/2, Y - H/2) / (W/2);
    Th = atan2(Y - H/2, X - W/2);
    edges = linspace(0, 1, 13);                 % 12 rings
    f = [];
    for c = 1:3
        ch = img(:, :, c);
        prof = zeros(1, 12);
        for k = 1:12
            prof(k) = mean(ch(R >= edges(k) & R < edges(k+1)));
        end
        f = [f, prof, diff(prof), ...
             prctile(ch(R < 0.5), [10 50 90]), std(ch(R < 0.5))];
    end

    % Angular sectors inside the disc region (green channel)
    g = img(:, :, 2);
    sectEdges = linspace(-pi, pi, 5);
    for s = 1:4
        m = R < 0.6 & Th >= sectEdges(s) & Th < sectEdges(s+1);
        f = [f, mean(g(m)), std(g(m))];
    end

    % Crude CDR: bright-region extents in red (disc) and green (cup)
    disc = img(:, :, 1) > multithresh(img(R < 0.9, 1)  == img(R < 0.9, 1), 1) * 0; %#ok<*NASGU>
end