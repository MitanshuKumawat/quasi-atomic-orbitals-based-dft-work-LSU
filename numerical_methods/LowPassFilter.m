% ===== VECTORIZED Smooth Filter =====
function filter = LowPassFilter(G_mag, G_cut, delta)
    filter = zeros(size(G_mag));
    G_del_diff = G_cut - delta;
    G_del_sum = G_cut + delta;
    
    % Region 1: Full pass (|G| ≤ Gcut - delta)
    idx1 = (G_mag <= G_del_diff);
    filter(idx1) = 1;
    
    % Region 2: Cosine transition (Gcut - delta < |G| < Gcut + delta)
    idx2 = (G_mag > G_del_diff) & (G_mag < G_del_sum);
    filter(idx2) = 0.5 * (1 + cos(pi * (G_mag(idx2) - G_del_diff) / (2*delta)));
end
