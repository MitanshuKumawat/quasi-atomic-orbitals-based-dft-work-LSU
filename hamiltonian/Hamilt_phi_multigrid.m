function [Hphi_t, Sphi_t] = Hamilt_phi_multigrid(phi, ibz, ispin)
% HAMILT_PHI_MULTIGRID - Multi-grid implementation of H|phi> and S|phi>
% 
% [Hphi_t, Sphi_t] = Hamilt_phi_multigrid(phi, ibz, ispin)
%
% Applies Hamiltonian and overlap operators to wavefunctions using multiple grid levels.
% phi(nWaves,nQOrbs,3) --> because it's for a QO (not all nQOrbs)
%
% Inputs:
%   phi   - Wavefunctions on multiple grid levels
%   ibz   - IBZ k-point index
%   ispin - Spin index
%
% Outputs:
%   Hphi_t - H|phi> on multiple grid levels
%   Sphi_t - S|phi> on multiple grid levels

global PW Atom Bloch Psp FFT3 FFT

nQOrbs = size(phi,2);

[nWaves, ~, ~] = size(phi);
Hphi_t = zeros(nWaves, nQOrbs, 3);
Sphi_t = zeros(nWaves, nQOrbs, 3);
AtomMax = Atom.AtomMax;

q = PW.PWIndex(:, :, ibz) + PW.IBZK(ibz,:);
q = q * Atom.G';

% ===== KINETIC ENERGY (DIAGONAL IN GRID LEVELS) =====
for i = 1:3
    Hphi_t(:,:,i) = 0.5 * sum(q.^2, 2) .* phi(:,:,i);
    Sphi_t(:,:,i) = phi(:,:,i);
end


% ===== Vloc3 =====
Vloc3 = cell(1,3);
for j = 1:3
    Vloc_cube = reshape(Bloch.Vloc(:,ispin),FFT.fftGrid);

    Vloc_center = fftshift(Vloc_cube);

    startIdx = floor((FFT3(j).fftGrid - FFT.fftGrid)/2) + 1;
    endIdx = startIdx + FFT.fftGrid - 1;

    Vloc_cube3 = zeros(FFT3(j).fftGrid);

    Vloc_cube3(startIdx(1):endIdx(1), ...
              startIdx(2):endIdx(2), ...
              startIdx(3):endIdx(3)) = Vloc_center;

    Vloc_std = ifftshift(Vloc_cube3);
    Vloc3{j} = Vloc_std(:);
end


% ===== LOCAL POTENTIAL (CROSS-LEVEL TERMS) =====
for j = 1:3 
    phi_j = phi(:,:,j);
    phiR_j = phiIFFT(phi_j, ibz, FFT3(j));
        
    for i = 1:3 
        %VphiR = Bloch.Vloc(:,i,ispin) .* phiR_j;
        %VphiR = Bloch.Vloc(:,ispin) .* phiR_j;
        VphiR = Vloc3{i} .* phiR_j;
        
        VphiK = phiFFT(VphiR, ibz, FFT3(i));
        Hphi_t(:,:,i) = Hphi_t(:,:,i) + filterK(VphiK, ibz, i);
        %Hphi_t(:,:,i) = Hphi_t(:,:,i) + VphiK;
    end
end

% ===== NON-LOCAL POTENTIAL (CROSS-LEVEL TERMS) =====
H_nonlocal = zeros(nWaves, nQOrbs, 3);
S_nonlocal = zeros(nWaves, nQOrbs, 3);

for j = 1:3
    phi_j = phi(:,:,j);
    
    for iatom = 1:AtomMax
        itype = Atom.Atom2Type(iatom);
        projMax = Psp{itype}.projMax;
        phase = Atom.Phase(:,iatom,ibz);
        % Distribute to grid levels with filtering
        for i = 1:3
            p_I = filterK(Psp{itype}.projK(:,:,ibz) .* phase,ibz,i);  % (nWaves,projMax)
        
            % Projector matrix elements (cross-level)
            proj = p_I' * phi_j;  % (projMax,1)
    
            H_proj = p_I * (Atom.D(1:projMax,1:projMax,iatom,ispin) * proj);
            S_proj = p_I * (Atom.Q(1:projMax,1:projMax,iatom,ispin) * proj);

            H_nonlocal(:,:,i) = H_nonlocal(:,:,i) + filterK(H_proj, ibz, i);
            S_nonlocal(:,:,i) = S_nonlocal(:,:,i) + filterK(S_proj, ibz, i);
        end
    end
end

Hphi_t = Hphi_t + H_nonlocal;
Sphi_t = Sphi_t + S_nonlocal;
end

