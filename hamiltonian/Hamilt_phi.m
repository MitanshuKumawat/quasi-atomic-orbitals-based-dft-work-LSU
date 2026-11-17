function [Hphi, Sphi, E, phi] = Hamilt_phi(phi, ibz, ispin, normalize)
% HAMILT_PHI - Applies Hamiltonian and overlap operators to wavefunctions
% 
% [Hphi, Sphi, E, phi] = Hamilt_phi(phi, ibz, ispin, normalize)
%
% Applies Hamiltonian and overlap operators to wavefunctions using vectorized operations.
% Optionally computes energy expectation values and normalizes results.
%
% Inputs:
%   phi       - Wavefunctions (nWaves x nQOrbs)
%   ibz       - IBZ k-point index
%   ispin     - Spin index
%   normalize - (Optional) If true, normalizes phi, Hphi, and Sphi
%
% Outputs:
%   Hphi    - H|phi> for all orbitals
%   Sphi    - S|phi> for all orbitals
%   E       - Energy expectation values for each orbital
%   phi     - Normalized wavefunctions (if normalize=true)

global PW Atom Bloch Psp FFT

% Check that n_electrons is less than 2*nQOrbs to prevent method failure
nQOrbs = size(phi, 2);
if isfield(Atom, 'n_electrons') && Atom.n_electrons >= 2*nQOrbs
    error('Hamilt_phi: n_electrons (%d) must be less than 2*nQOrbs (%d) to prevent method failure', Atom.n_electrons, 2*nQOrbs);
end

AtomMax = Atom.AtomMax;

% Kinetic Energy
q = PW.PWIndex(:, :, ibz) + PW.IBZK(ibz,:);
q = q * Atom.G';

HSQDTM = 13.605826 * 0.529177249^2; % Same as VASP's constant in eV·Å²
Hphi = HSQDTM*sum(q.^2,2).*phi;     % Correct: T = (ℏ²/2m)|k+G|²
Sphi = phi;

% Local contributions
tmp = phiIFFT(phi, ibz, FFT);
% Vloc(nPoints, nSpins) --> (nPoints, 1) for this spin
tmp = Bloch.Vloc(:,ispin).*tmp; % Vloc_phi_r(nPoints, nQOrbs)
Hphi = Hphi + phiFFT(tmp, ibz, FFT);

% Non-local contributions
for iatom = 1:AtomMax
    itype = Atom.Atom2Type(iatom);
    projMax = Psp{itype}.projMax;
    temp2 = Atom.Phase(:,iatom,ibz);
    temp1 = Psp{itype}.projK(:,:,ibz); %(nWaves,projMax)
    p_I = temp1.*temp2;

    proj = p_I'*phi; % proj(projMax,nQOrbs) = p_jI_G(nWaves,projMax,ibz,ispin)'*phi_mu(nWaves,nQOrbs,ibz,ispin)
  
    Hphi = Hphi + p_I*(Atom.D(1:projMax,1:projMax,iatom,ispin)*proj);
    Sphi = Sphi + p_I*(Atom.Q(1:projMax,1:projMax,iatom,ispin)*proj);
end

% Compute energy E for each column of phi
% Supports single-column (vector) or multi-column input
num = real(sum(conj(phi).*Hphi,1));
den = real(sum(conj(phi).*Sphi,1));
% Validate denominator before division
if any(~isfinite(den) | den <= 0)
    error('Hamilt_phi: non-positive or non-finite <psi|S|psi> encountered (min=%.2e)', min(den));
end
% If phi is a single vector, E is scalar; otherwise diagonal elements
E = num ./ den;

% Optional normalization of phi, Hphi, Sphi (and return normalized phi)
if nargin >= 4 && ~isempty(normalize) && normalize
    c = sqrt(den);
    % Normalize columns
    phi  = phi  ./ c;
    Hphi = Hphi ./ c;
    Sphi = Sphi ./ c;
    % Verify S-based normalization per column
    s_after = real(sum(conj(phi).*Sphi,1));
    if max(abs(s_after - 1)) > 1e-10
        error('Hamilt_phi: S-normalization failed, max error=%.2e', max(abs(s_after - 1)));
    end
    % Check that the Euclidean norm is within a factor of 10 of the S-based norm
    eucl_norm = sum(abs(phi).^2,1);
    s_norm = s_after;
    ratio = eucl_norm ./ s_norm;
    if any(ratio > 10 | ratio < 0.1)
        warning('Hamilt_phi: Euclidean norm and S-based norm differ by more than a factor of 10 (min ratio=%.2e, max ratio=%.2e)', min(ratio), max(ratio));
    end
    % E is scale-invariant; keep as is
end
end
