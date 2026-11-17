function [rho, rho1, rho2, rho3, r] = ProjectQOToMultiFFT(phi_vasp, ibz, iatom)
    global FFT1 FFT2 FFT3 FFT Atom PW;
    % Input: 
    %   phi_vasp: (nWaves x nQOrbs), from DIIS output
    %   FFT: struct with fields level1, level2, level3
    %   ibz: IBZ index used in phiIFFT
    %   n: which orbital to project (e.g., 1 to nQOrbs)

    % Output:
    %   rho1, rho2, rho3: radial densities for each level
    %   r: corresponding radial distances

    % Pre-filtration setup
    fact = 2;
    Gmax = max(PW.nGrid./vecnorm(Atom.H,2,2).');
    G_cut1 = Gmax/3;
    G_cut2 = 2*Gmax/3;
    delta = fact*max(1./vecnorm(Atom.H,2,2).');

    q = PW.PWIndex(:, :, ibz) + PW.IBZK(ibz,:);
    q = q * Atom.G';
    G_mag = sqrt(sum(q.^2, 2));  % (nWaves, 1) - magnitudes
    fprintf("Gmax = %d, delta = %d \n", Gmax, delta);

    filter3 = LowPassFilter(G_mag,G_cut1,delta);  % Coarse grid (low freq)
    filter1 = 1 - LowPassFilter(G_mag,G_cut2,delta);  % Fine grid (high freq)
    filter2 = LowPassFilter(G_mag,G_cut2,delta) - LowPassFilter2(G_mag,G_cut1,delta);  % Mid grid

    phi3 = phi_vasp .* filter3; %coarse grid
    phi2 = phi_vasp .* filter2; %mid grid
    phi1 = phi_vasp .* filter1; %fine grid

    reconstruct = phi1 + phi2 + phi3;
    reconstruct_error = norm(phi_vasp - reconstruct) / norm(phi_vasp);
    
    fprintf('Reconstruction error: %.6f%%\n', reconstruct_error * 100);

   
    %% ===== Plot RECIPROCAL SPACE =====
    %[sorted_G, idx] = sort(G_mag);
   %
    %phi_sorted     = abs(phi_vasp(idx)).^2;
    %phi1_sorted    = abs(phi1(idx)).^2;
    %phi2_sorted    = abs(phi2(idx)).^2;
    %phi3_sorted    = abs(phi3(idx)).^2;
    %
    %figure('Position', [100, 100, 1200, 800]);
    %subplot(2, 3, 1);
    %plot(sorted_G, phi_sorted, 'k-', 'DisplayName', 'Unfiltered');
    %xlabel('|G| (Å^{-1})');
    %ylabel('|\phi(G)|^2');
    %title('Reciprocal Space Density');
    %legend;
    %subplot(2, 3, 2);
    %plot(sorted_G, phi1_sorted, 'r-', 'DisplayName', 'Level 1');
    %xlabel('|G| (Å^{-1})');
    %ylabel('|\phi(G)|^2');
    %title('Reciprocal Space Density');
    %legend;
    %subplot(2, 3, 3);
    %plot(sorted_G, phi2_sorted, 'g-', 'DisplayName', 'Level 2');
    %xlabel('|G| (Å^{-1})');
    %ylabel('|\phi(G)|^2');
    %title('Reciprocal Space Density');
    %legend;
    %subplot(2, 3, 4);
    %plot(sorted_G, phi3_sorted, 'b-', 'DisplayName', 'Level 3');
    %xlabel('|G| (Å^{-1})');
    %ylabel('|\phi(G)|^2');
    %title('Reciprocal Space Density');
    %legend;
    %grid on;

    %% coversion to real space
    phi_r = phiIFFT(phi_vasp,ibz,FFT);
    %phi1 = phiIFFT(phi1,ibz,FFT1);
    phi1 = phiIFFT(phi1,ibz,FFT1);
    phi2 = phiIFFT(phi2,ibz,FFT2);
    phi3 = phiIFFT(phi3,ibz,FFT3);

    % === Step 5: Compute spherical average densities ===
    coor_atm = Atom.S(iatom,:);
    % Assume same radial bins for all three levels for now
    %r = linspace(0, 1, numel(rho1));  % normalize radius for plotting

    % ===== 4. Real Space Radial Density Plot =====
    % After computing real-space wavefunctions:
    %[r, density] = RadialDensity(phi_r, FFT, coor_atm);
    %[r1, density1] = RadialDensity(phi1, FFT1, coor_atm);
    %[r2, density2] = RadialDensity(phi2, FFT2, coor_atm);
    %[r3, density3] = RadialDensity(phi3, FFT3, coor_atm);
    %
    %% Sort by radial distance for cleaner plotting
    %[r_sorted, idx] = sort(r);
    %density_sorted = density(idx);
    %
    %[r1_sorted, idx1] = sort(r1);
    %density1_sorted = density1(idx1);
    %
    %[r2_sorted, idx2] = sort(r2);
    %density2_sorted = density2(idx2);
    %
    %[r3_sorted, idx3] = sort(r3);
    %density3_sorted = density3(idx3);
    %
    %% Create plot
    %figure('Position', [100, 100, 1000, 800]);
    %
    %% Unfiltered
    %subplot(2,2,1);
    %scatter(r_sorted, density_sorted, 1, 'k.');
    %xlabel('Radial Distance (Å)');
    %ylabel('|\phi(r)|^2');
    %title('Total Real Space Density');
    %%set(gca, 'YScale', 'log');  % Logarithmic scale recommended
    %grid on;
    %
    %% Level 1 (High-Freq)
    %subplot(2,2,2);
    %scatter(r1_sorted, density1_sorted, 1, 'r.');
    %xlabel('Radial Distance (Å)');
    %ylabel('|\phi(r)|^2');
    %title('Level 1 (Fine Grid)');
    %%set(gca, 'YScale', 'log');
    %grid on;
    %
    %% Level 2 (Mid-Freq)
    %subplot(2,2,3);
    %scatter(r2_sorted, density2_sorted, 1, 'g.');
    %xlabel('Radial Distance (Å)');
    %ylabel('|\phi(r)|^2');
    %title('Level 2 (Mid Grid)');
    %%set(gca, 'YScale', 'log');
    %grid on;
    %
    %% Level 3 (Low-Freq)
    %subplot(2,2,4);
    %scatter(r3_sorted, density3_sorted, 1, 'b.');
    %xlabel('Radial Distance (Å)');
    %ylabel('|\phi(r)|^2');
    %title('Level 3 (Coarse Grid)');
    %%set(gca, 'YScale', 'log');
    %grid on;


    % Convert to 3D arrays
    phi_r_3D = reshape(phi_r, size(FFT.GridX));
    phi1_3D = reshape(phi1, size(FFT1.GridX));
    phi2_3D = reshape(phi2, size(FFT2.GridX));
    phi3_3D = reshape(phi3, size(FFT3.GridX));

    % Calculate global maximum density value
    global_max = max([abs(phi_r_3D(:)).^2; 
                     abs(phi1_3D(:)).^2;
                     abs(phi2_3D(:)).^2;
                     abs(phi3_3D(:)).^2]);
                 
    % Set consistent color limits (log scale recommended)
    clim_min = 1e-6 * global_max;  % Adjust factor as needed
    clim_max = global_max;
    
    % Generate contour plots
    PlotContourSlices(phi_r_3D, FFT.GridX, FFT.GridY, FFT.GridZ, coor_atm, 'Total Wavefunction',clim_min, clim_max);
    PlotContourSlices(phi1_3D, FFT1.GridX, FFT1.GridY, FFT1.GridZ, coor_atm, 'Level 1 (Fine)',clim_min, clim_max);
    PlotContourSlices(phi2_3D, FFT2.GridX, FFT2.GridY, FFT2.GridZ, coor_atm, 'Level 2 (Mid)',clim_min, clim_max);
    PlotContourSlices(phi3_3D, FFT3.GridX, FFT3.GridY, FFT3.GridZ, coor_atm, 'Level 3 (Coarse)',clim_min, clim_max);

end
