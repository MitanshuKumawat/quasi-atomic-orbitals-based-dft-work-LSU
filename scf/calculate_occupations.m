function occ = calculate_occupations(eigenvalues, n_electrons, sigma, k_weights)
% CALCULATE_OCCUPATIONS - Calculate electronic occupations using Gaussian smearing
%
% occ = calculate_occupations(eigenvalues, n_electrons, sigma, k_weights)
%
% Calculates electronic occupation numbers using Gaussian smearing method,
% which is commonly used in electronic structure calculations to handle
% metallic systems. This function determines a single Fermi level for the
% entire system and returns occupations for all states.
%
% Inputs:
%   eigenvalues  - Array of eigenvalues (energies) in eV with dimensions (nQOrbs, nIBZKs, nSpins)
%   n_electrons  - Total number of electrons to distribute
%   sigma        - Smearing width in eV
%   k_weights    - Array of k-point weights with dimensions (nIBZKs)
%
% Output:
%   occ - Array of occupation numbers with the same dimensions as eigenvalues
%
% Note: Spin weighting is handled automatically based on the dimensions of eigenvalues.
%       For non-spin-polarized calculations (nSpins=1), each state can hold 2 electrons.
%       For spin-polarized calculations (nSpins=2), each state can hold 1 electron.

% Find min and max eigenvalues without sorting
e_min = min(eigenvalues(:));
e_max = max(eigenvalues(:));

% Set initial Fermi energy window with padding (similar to VASP's DENSTA_CSTR)
e_add = (e_max - e_min) / 2;
e_add = max(e_add, 10 * sigma);
e_min = e_min - e_add;
e_max = e_max + e_add;

% Initial guess for Fermi energy: midpoint of window
e_fermi_init = (e_min + e_max) / 2;

% Function to calculate total electron count for a given Fermi energy
function n_total = electron_count(e_fermi)
    % Initialize weighted electron count
    n_total = 0;
    
    % Get dimensions
    [nQOrbs, nIBZKs, nSpins] = size(eigenvalues);
    
    % Determine spin weighting factor
    % For non-spin-polarized (nSpins=1): spin_factor = 2
    % For spin-polarized (nSpins=2): spin_factor = 1
    spin_factor = 2/nSpins;  % Gives 2 when nSpins=1, 1 when nSpins=2
    
    % Loop over spins and k-points to apply weights
    for ispin = 1:nSpins
        for ibz = 1:nIBZKs
            % Get eigenvalues for this k-point and spin
            evals_k = eigenvalues(:, ibz, ispin);
            
            % Calculate occupations for this k-point
            arg = (evals_k - e_fermi) / sigma;
            occ_k = 0.5 * erfc(arg);
            
            % Add weighted contribution to total electron count
            % Apply both k-point weight and spin factor
            n_total = n_total + k_weights(ibz) * sum(occ_k) * spin_factor;
        end
    end
end

% Function to minimize: difference between calculated and target electron count
function diff = electron_diff(e_fermi)
    diff = electron_count(e_fermi) - n_electrons;
end

% Use MATLAB's built-in root finder to find the Fermi energy
% First check if we need to search (sometimes all bands are fully occupied or empty)
elect_min = electron_count(e_min);
if abs(elect_min - n_electrons) < 1e-10
    e_fermi = e_min;
else
    elect_max = electron_count(e_max);
    if abs(elect_max - n_electrons) < 1e-10
        e_fermi = e_max;
    else
        % Use fzero to find Fermi energy where electron_diff = 0
        options = optimset('TolX', 1e-10, 'Display', 'off');
        e_fermi = fzero(@electron_diff, [e_min, e_max], options);
    end
end

% Calculate final occupations with the determined Fermi energy
% Initialize output array with same dimensions as eigenvalues
occ = zeros(size(eigenvalues));

% Apply Fermi function to each k-point and spin separately
[nQOrbs, nIBZKs, nSpins] = size(eigenvalues);
for ispin = 1:nSpins
    for ibz = 1:nIBZKs
        % Get eigenvalues for this k-point and spin
        evals_k = eigenvalues(:, ibz, ispin);
        
        % Calculate occupations using Fermi function
        arg = (evals_k - e_fermi) / sigma;
        occ(:, ibz, ispin) = 0.5 * erfc(arg);
    end
end

end
