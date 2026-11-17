function filtered = filterK(phi, ibz, level)
    global PW Atom
    fact = 2;
    Gmax = max(PW.nGrid./vecnorm(Atom.H,2,2).');
    G_cut1 = Gmax/3;        % Coarse grid cutoff
    G_cut2 = 2*Gmax/3;      % Mid grid cutoff
    delta = fact * max(1./vecnorm(Atom.H,2,2).');
    
    q = PW.PWIndex(:, :, ibz) + PW.IBZK(ibz,:); 
    q = q * Atom.G';
    G_mag = sqrt(sum(q.^2, 2));  % (nWaves, 1) - magnitudes
    
    coarse = LowPassFilter(G_mag, G_cut1, delta);
    mid = LowPassFilter(G_mag, G_cut2, delta) - coarse;
    fine = 1 - LowPassFilter(G_mag, G_cut2, delta);
    
    switch level
        case 1 
            filter = fine;
        case 2  
            filter = mid;
        case 3 
            filter = coarse;
    end
    
    filtered = phi .* filter;
end