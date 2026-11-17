%function [H,S] = H_S(phi,ibz,ispin)
%% phi(nWaves,nQOrbs,nIBZKs,nSpins)
%
%nQOrbs = size(phi,2);
%
%H = zeros(nQOrbs, nQOrbs);
%S = zeros(nQOrbs, nQOrbs);
%
%for nu = 1:nQOrbs
%    [Hphi, Sphi] = Hamilt_phi(phi(:,nu),ibz,ispin);
%    H(:,nu) = phi'*Hphi;
%    S(:,nu) = phi'*Sphi;
%end
%
%H = (H+H')/2;
%S = (S+S')/2;
%
%end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [H,S] = H_S(phi,ibz,ispin)
% phi_mu(nWaves,nQOrbs,nIBZKs,nSpins)
[Hphi, Sphi] = Hamilt_phi(phi,ibz,ispin);
H = phi'*Hphi;
S = phi'*Sphi;

% Check Hermiticity before symmetrizing
H_hermitian_error = abs(H-H')./(H+H');
S_hermitian_error = abs(S-S')./(S+S');

if max(max(H_hermitian_error), max(S_hermitian_error)) > 1e-10
    error('H_S: Non-Hermitian matrices detected before symmetrization: H_err=%.2e, S_err=%.2e', ...
        max(max(H_hermitian_error)), max(max(S_hermitian_error)));
end

% Symmetrize matrices
H = (H+H')/2;
S = (S+S')/2;
end



%%%%%%%%%%%%%%%%%%%%%%%%%% MULTI-GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%function [H,S] = H_S(phi,ibz,ispin)
%    [Hphi, Sphi] = Hamilt_phi(phi,ibz,ispin);
%    nQOrbs = size(phi,2);
%    
%    % Initialize with cross-level terms
%    H = zeros(nQOrbs);
%    S = zeros(nQOrbs);
%    
%    % Include all cross terms (9 combinations)
%    for i = 1:3
%        for j = 1:3
%            H = H + phi(:,:,i)' * Hphi(:,:,j);
%            S = S + phi(:,:,i)' * Sphi(:,:,j);
%        end
%    end
%    
%    % Symmetrize
%    H = (H + H')/2;
%    S = (S + S')/2;
%end