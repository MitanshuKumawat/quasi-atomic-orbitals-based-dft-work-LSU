function [Hphi, Sphi] = Hamilt_phi_loopbased(phi, ibz, ispin)
% HAMILT_PHI_LOOPBASED - Loop-based implementation of H|phi> and S|phi>
% 
% [Hphi, Sphi] = Hamilt_phi_loopbased(phi, ibz, ispin)
%
% Applies Hamiltonian and overlap operators to wavefunctions using a loop-based approach
% that processes one orbital at a time.
%
% Inputs:
%   phi   - Wavefunctions (nWaves x nQOrbs)
%   ibz   - IBZ k-point index
%   ispin - Spin index
%
% Outputs:
%   Hphi - H|phi> for all orbitals
%   Sphi - S|phi> for all orbitals

global PW Atom Bloch Psp FFT

AtomMax = Atom.AtomMax;
nQOrbs = size(phi, 2);  % Number of orbitals

% Initialize outputs
Hphi = zeros(size(phi));
Sphi = phi;  % Start with identity contribution for S

% Kinetic Energy (column-wise, no coupling)
q = PW.PWIndex(:, :, ibz) + PW.IBZK(ibz,:);
q = q * Atom.G';
Ekin_G = 0.5 * sum(q.^2, 2);
Hphi = Ekin_G .* phi;  % Column-wise multiplication

% Local contributions (column-wise, no coupling)
for iorb = 1:nQOrbs
    % Process one orbital at a time
    tmp = phiIFFT(phi(:, iorb), ibz, FFT);
    tmp = Bloch.Vloc(:, ispin) .* tmp;
    Hphi(:, iorb) = Hphi(:, iorb) + phiFFT(tmp, ibz, FFT);
end

% Non-local contributions (column-wise, no coupling)
for iatom = 1:AtomMax
    itype = Atom.Atom2Type(iatom);
    projMax = Psp{itype}.projMax;
    temp2 = Atom.Phase(:, iatom, ibz);
    temp1 = Psp{itype}.projK(:, :, ibz);  % (nWaves, projMax)
    p_I = temp1 .* temp2;

    for iorb = 1:nQOrbs
        % Process one orbital at a time
        phi_orb = phi(:, iorb);  % Extract single orbital
        
        % Projection for THIS orbital only
        proj = p_I' * phi_orb;  % (projMax x 1)
        
        % Apply non-local operator to THIS orbital only
        H_nonlocal = p_I * (Atom.D(1:projMax, 1:projMax, iatom, ispin) * proj);
        S_nonlocal = p_I * (Atom.Q(1:projMax, 1:projMax, iatom, ispin) * proj);
        
        Hphi(:, iorb) = Hphi(:, iorb) + H_nonlocal;
        Sphi(:, iorb) = Sphi(:, iorb) + S_nonlocal;
    end
end
end

