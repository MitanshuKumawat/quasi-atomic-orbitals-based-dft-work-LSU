function [psi_all, E_all] = RMM_DIIS_band(params)

% RMM_DIIS_band - RMM-DIIS orbital optimization for electronic structure
%
% [psi_all, E_all] = RMM_DIIS_band(params)
%
% Implements the Residual Minimization Method with Direct Inversion in Iterative
% Subspace (RMM-DIIS) algorithm for optimizing electronic wavefunctions.
% This version uses per-band history storage and implements convergence criteria
% matching the VASP implementation.
%
% Key algorithm steps:
% 1. Subspace rotation: Diagonalize H/S to get initial eigenpairs (C, E0)
% 2. For each band n:
%    a. Trial step (i=1): Compute search direction s = K*R, find optimal step size α
%       using quadratic model, update ψ = ψ - α*s
%    b. DIIS steps (i>1): Build matrices A = R'*s and B = ψ'*S*ψ from history,
%       solve generalized eigenvalue problem A*b = λ*B*b, update using
%       ψ = (ψ_diis - α*s_diis)*b
%    c. Check convergence using energy change, first-order criterion, and occupation
% 3. Monitor overall convergence using total energy change and weighted energy sum
%
% Parameters in 'params' struct control convergence thresholds, step sizes,
% maximum iterations, and logging behavior.
%
% Inputs:
%   params - Structure containing algorithm parameters (see set_params function)
%
% Outputs:
%   psi_all - Optimized wavefunctions (nWaves, nQOrbs, nIBZKs, nSpins)
%   E_all   - Eigenvalues for each band and k-point (nQOrbs, nIBZKs)

global PW QMB Bloch

% Initialize parameters with defaults if not provided
if nargin < 1 || isempty(params)
    params = struct();
end

% Set up parameters with defaults
% Algorithm control parameters
params.Nmax = set_params(params, 'Nmax', 4);   % Maximum DIIS history size
params.max_inner_iters = set_params(params, 'max_inner_iters', 20);   % Maximum number of outer iterations
params.tol = set_params(params, 'tol', 1e-7);   % General convergence tolerance

% Line search parameters
params.line_search = struct();
params.line_search.alpha_min = set_params(params, 'alpha_min', 0.1);   % Minimum step size for line search
params.line_search.alpha_max = set_params(params, 'alpha_max', 1.0);   % Maximum step size for line search

% Logging parameters
params.verbose = set_params(params, 'verbose', true);   % Enable/disable verbose output
params.log_every = set_params(params, 'log_every', 1);   % Log frequency (iterations)

% Convergence control parameters (matching VASP)
params.convergence = struct();
params.convergence.energy_tol = set_params(params, 'energy_tol', 1e-6);   % Total energy change tolerance
params.convergence.dE_tol = set_params(params, 'dE_tol', 1e-8);   % Per-band energy change tolerance
params.convergence.dE1_scale = set_params(params, 'dE1_scale', 0.5);   % First-order convergence criterion scale
params.convergence.occ_min = set_params(params, 'occ_min', 1e-4);   % Minimum occupation for convergence checks

% Smearing parameters for occupation calculation
params.smearing = struct();
params.smearing.sigma = set_params(params, 'sigma', 0.2);   % Gaussian smearing width (eV)

% No need to extract parameters since we're passing the entire params struct

% Get dimensions from global data structures
nSpins = PW.nSpins;
nIBZKs = PW.nIBZKs;
nQOrbs = QMB.nQOrbs;

E_all = zeros(nQOrbs, nIBZKs, nSpins);

% Calculate spin weighting factor
spin_factor = 2/nSpins;  % Gives 2 when nSpins=1, 1 when nSpins=2

% Calculate total number of electrons as sum over (nQOrbs, nSpins) and weighted by IBZKWeight
n_electrons = sum(Bloch.Occupation, [1,3]) * PW.IBZKWeight;

psi_all = permute(QMB.aorbK(:, :, :, 1), [1 3 2 4]);

% Initialize convergence data structure
conv_data = struct();
conv_data.iter = 0;
conv_data.Etot_prev = inf;
conv_data.converged = false;
conv_data.dEtot = 0;
conv_data.dE_sum = 0;
conv_data.RMS = 0;
conv_data.Etot = 0;

% Compute E0_all by replicating Bloch.EigenValues(:,1,1) to match (nQOrbs, nIBZKs)
E0_all = repmat(Bloch.EigenValues(:,1,1), [1, nIBZKs, nSpins]);

% Compute occupancies using calculate_occupancies
occ_all = calculate_occupations(E0_all, n_electrons, params.smearing.sigma, PW.IBZKWeight);

% Compute total energy: sum over all bands and k-points, weighted by occupation and IBZKWeight
Etot = spin_factor * sum(E0_all .* occ_all,1) * PW.IBZKWeight;

% Print the total energy
fprintf('Total VASP energy (from Bloch.EigenValues): %.10f eV\n', Etot);


% Main iteration loop
for iter = 1 : params.max_inner_iters
    conv_data.iter = iter;
    
    % Check wavefunction state before eigenvalue calculation
    if iter > 1
        % Check wavefunction state before eigenvalue calculation
        psi_norm = norm(psi_all(:));
        
        % Check for NaN/Inf
        if ~isfinite(psi_norm)
            error('Non-finite wavefunctions detected at iteration %d: norm=%.2e', iter, psi_norm);
        end
    end
    
    % ===== PHASE 1: Calculate eigenvalues for all ibz-points and spins =====
    [psi_all, E0_all, occ_all] = calculate_eigenvalues_and_occupations(psi_all, n_electrons, params);
    
    % ===== PHASE 2: Optimize wavefunctions using calculated occupations =====
    [psi_all, E_all, conv_data] = optimize_wavefunctions(psi_all, E0_all, occ_all, params, conv_data);
    
    % Check wavefunction state after optimization
    psi_norm = norm(psi_all(:));
    
    % Check for NaN/Inf
    if ~isfinite(psi_norm)
        error('Non-finite wavefunctions detected after optimization at iteration %d: norm=%.2e', iter, psi_norm);
    end
    
    % Check convergence and report progress
    conv_data = check_convergence(conv_data, params);
    if conv_data.converged
        break;
    end
    
    conv_data.Etot_prev = conv_data.Etot;
end

end  % End of main function RMM_DIIS_band

% -----------------------------------------------------------------------------
% CALCULATE_EIGENVALUES_AND_OCCUPATIONS - Calculate eigenvalues and occupations
%
% [E0_all, occ_all] = calculate_eigenvalues_and_occupations(psi_all, n_electrons, params)
%
% Calculates eigenvalues for all k-points and spins by diagonalizing the
% Hamiltonian and overlap matrices, then computes occupations.
%
% Inputs:
%   psi_all     - Wavefunctions array (nWaves, nQOrbs, nIBZKs, nSpins)
%   n_electrons - Total number of electrons
%   params      - Parameter struct containing smearing parameters
%
% Outputs:
%   E0_all  - Eigenvalues for all bands, k-points, and spins (nQOrbs, nIBZKs, nSpins)
%   occ_all - Occupations for all bands, k-points, and spins (nQOrbs, nIBZKs, nSpins)
% -----------------------------------------------------------------------------
function [psi_all, E0_all, occ_all] = calculate_eigenvalues_and_occupations(psi_all, n_electrons, params)
    global PW
    
    % Get dimensions
    [~, nQOrbs, nIBZKs, nSpins] = size(psi_all);
    
    % Initialize output arrays
    E0_all = zeros(nQOrbs, nIBZKs, nSpins);
    
    % Calculate eigenvalues for all k-points and spins
    for ispin = 1:nSpins
        for ibz = 1:nIBZKs
            % In debug mode, only compute for ibz=1, ispin=1; others copy and continue
            if params.debug && (ibz > 1 || ispin > 1)
                % Copy eigenvalues and rotated psi from reference (ibz=1, ispin=1)
                E0_all(:, ibz, ispin) = E0_all(:, 1, 1);
                psi_all(:, :, ibz, ispin) = psi_all(:, :, 1, 1);
                continue;
            end
            % Get current wavefunction for this ibz-point and spin
            psi_ibz = psi_all(:, :, ibz, ispin);
            
            % Check wavefunction before H_S
            psi_norm = norm(psi_ibz, 'fro');
            if ~isfinite(psi_norm) || psi_norm < 1e-10 || psi_norm > 1e10
                error('Invalid wavefunction detected before H_S: norm = %.2e', psi_norm);
            end
            
            % Subspace rotation
            [H_ibz, S_ibz] = H_S(psi_ibz, ibz, ispin);
            
            % Check matrices after H_S
            H_norm = norm(H_ibz, 'fro');
            S_norm = norm(S_ibz, 'fro');
            if ~isfinite(H_norm) || ~isfinite(S_norm)
                error('Non-finite matrices from H_S: H_norm = %.2e, S_norm = %.2e', H_norm, S_norm);
            end
            
            [C, E0] = eig(H_ibz, S_ibz);
            
            % Check Hermiticity of matrices (after H_S symmetrization)
            H_hermitian_error = abs(H_ibz-H_ibz')./(H_ibz+H_ibz');
            S_hermitian_error = abs(S_ibz-S_ibz')./(S_ibz+S_ibz');
            
            if max(max(H_hermitian_error), max(S_hermitian_error)) > 1e-10
                error('Non-Hermitian matrices detected after H_S: H_err=%.2e, S_err=%.2e', ...
                    max(max(H_hermitian_error)), max(max(S_hermitian_error)));
            end
            
            % Verify eigenvalues are real (exactly, not just small imaginary part)
            E0_diag = diag(E0);
            if any(~isreal(E0_diag))
                max_imag = max(abs(imag(E0_diag)));
                error('Non-real eigenvalues detected (max imag = %.2e). This indicates a problem with Hermiticity of H or S.', max_imag);
            end
            
            % Update wavefunction with rotation (return rotated psi)
            psi_all(:, :, ibz, ispin) = psi_ibz * C;
            
            % Store eigenvalues
            E0_all(:, ibz, ispin) = diag(E0);
        end
    end
    
    % Calculate occupations for all states together
    occ_all = calculate_occupations(E0_all, n_electrons, params.smearing.sigma, PW.IBZKWeight);
end

% -----------------------------------------------------------------------------
% OPTIMIZE_WAVEFUNCTIONS - Optimize wavefunctions using RMM-DIIS
%
% [psi_all, E_all, conv_data] = optimize_wavefunctions(psi_all, E0_all, occ_all, params, conv_data)
%
% Optimizes wavefunctions for all k-points and spins using the RMM-DIIS method.
%
% Inputs:
%   psi_all   - Current wavefunctions (nWaves, nQOrbs, nIBZKs, nSpins)
%   E0_all    - Initial eigenvalues (nQOrbs, nIBZKs, nSpins)
%   occ_all   - Occupations (nQOrbs, nIBZKs, nSpins)
%   params    - Algorithm parameters
%   conv_data - Convergence data structure
%
% Outputs:
%   psi_all   - Updated wavefunctions
%   E_all     - Updated eigenvalues
%   conv_data - Updated convergence data structure with Etot, dE_sum, RMS
% -----------------------------------------------------------------------------
function [psi_all, E_all, conv_data] = optimize_wavefunctions(psi_all, E0_all, occ_all, params, conv_data)
    global PW
    
    % Get dimensions
    [~, nQOrbs, nIBZKs, nSpins] = size(psi_all);
    
    % Initialize output arrays and metrics
    E_all = zeros(nQOrbs, nIBZKs, nSpins);
    conv_data.RMS = 0;
    conv_data.Etot = 0;
    conv_data.dE_sum = 0;
    
    % Process all k-points and spins
    for ispin = 1:nSpins
        for ibz = 1:nIBZKs
            % In debug mode, only compute for ibz=1, ispin=1; others copy and continue
            if params.debug && (ibz > 1 || ispin > 1)
                psi_all(:, :, ibz, ispin) = psi_all(:, :, 1, 1);
                E_all(:, ibz, ispin) = E_all(:, 1, 1);
                conv_data.RMS = conv_data.RMS; % no-op to keep structure unchanged
                continue;
            end
            % Get current data for this ibz-point and spin
            psi_ibz = psi_all(:, :, ibz, ispin);
            E0 = E0_all(:, ibz, ispin);
            occ = occ_all(:, ibz, ispin);
            
            % Construct simple kinetic-energy preconditioner K(G)
            K = build_preconditioner(psi_ibz, ibz);
    
            % Process each band independently to reuse Hpsi/Spsi between iterates
            [psi_ibz, E, RMS_sum] = optimize_bands(psi_ibz, E0, occ, K, ibz, ispin, params);
            
            % Update convergence metrics
            conv_data.RMS = conv_data.RMS + RMS_sum;
            
            % Store updated wavefunctions and energies
            psi_all(:, :, ibz, ispin) = psi_ibz;
            E_all(:,ibz, ispin) = E;
            
        end
    end
    % Define spin_factor (2 for nSpins=1, 1 for nSpins=2)
    spin_factor = 2 / PW.nSpins;
    conv_data.Etot = spin_factor * sum(E_all .* occ_all,1) * PW.IBZKWeight;
    conv_data.dE_sum = spin_factor * sum((E_all-E0_all) .* occ_all,1) * PW.IBZKWeight;
end

% -----------------------------------------------------------------------------
% BUILD_PRECONDITIONER - Build kinetic energy preconditioner
%
% K = build_preconditioner(psi_ibz, ibz)
%
% Constructs a simple kinetic-energy preconditioner for RMM-DIIS.
%
% Inputs:
%   psi_ibz - Wavefunctions for a specific k-point and spin
%   ibz     - IBZ k-point index
%
% Output:
%   K - Preconditioner array (nWaves, nQOrbs)
% -----------------------------------------------------------------------------
function K = build_preconditioner(psi_ibz, ibz)
    global PW Atom
    
    [nWaves, nQOrbs] = size(psi_ibz);
    
    % Calculate kinetic energy
    q = PW.PWIndex(:, :, ibz) + PW.IBZK(ibz,:);
    q = q * Atom.G';
    Ekin_G = 0.5*sum(q.^2,2); % Kinetic Energy
    
    % Build preconditioner for all bands at once using array operations
    Ekin_n = sum(abs(psi_ibz).^2 .* Ekin_G, 1);
    
    x = Ekin_G * (1 ./ Ekin_n);
    x2 = 27 + x .* (18 + x .* (12 + 8*x));
    K = 2 * x2 ./ (x2 + 16*x.^4) ./ (ones(nWaves, 1) * Ekin_n);

end

% -----------------------------------------------------------------------------
% OPTIMIZE_BANDS - Optimize individual bands using RMM-DIIS
%
% [psi_ibz, E, RMS_sum] = optimize_bands(psi_ibz, E0, occ, K, ibz, ispin, params)
%
% Optimizes individual bands for a specific k-point and spin.
%
% Inputs:
%   psi_ibz - Wavefunctions for a specific k-point and spin
%   E0      - Initial eigenvalues
%   occ     - Occupations
%   K       - Preconditioner
%   ibz     - IBZ k-point index
%   ispin   - Spin index
%   params  - Algorithm parameters
%
% Outputs:
%   psi_ibz - Updated wavefunctions
%   E       - Updated eigenvalues
%   RMS_sum - RMS residual sum (occupation weighted)
% -----------------------------------------------------------------------------
function [psi_ibz, E, RMS_sum] = optimize_bands(psi_ibz, E0, occ, K, ibz, ispin, params)
    [nWaves, nQOrbs] = size(psi_ibz);
    E = zeros(nQOrbs, 1);
    RMS_sum = 0;
    
    for n = 1:nQOrbs
        % Initialize per-band DIIS history arrays
        % Each column i stores data from the i-th DIIS iteration
        psi_diis  = zeros(nWaves, params.Nmax);  % Wavefunctions
        R_diis    = zeros(nWaves, params.Nmax);  % Residuals
        s_diis    = zeros(nWaves, params.Nmax);  % Search directions
        Spsi_diis = zeros(nWaves, params.Nmax);  % Overlap operator applied to wavefunctions

        % Current band state and initial Hpsi/Spsi
        psi = psi_ibz(:,n);
        [Hpsi, Spsi] = Hamilt_phi(psi, ibz, ispin);

        for i = 1:params.Nmax
            % Build residual/search from current Hpsi/Spsi for this band
            R = Hpsi - Spsi * E0(n);
            
            % Calculate search direction
            K_band = K(:,n);
            s = K_band .* R;
            
            % Update first-order estimate for convergence check (every iteration)
            dE1 = real(R' * s);

            % Record iterate i in per-band history (for DIIS build)
            psi_diis(:,i)  = psi;
            R_diis(:,i)    = R;
            s_diis(:,i)    = s;
            Spsi_diis(:,i) = Spsi;

            if i == 1
                % Trial step (i=1): compute E_trial from psi - s, then alpha
                psi_trial = psi - s;
                [~, ~, E_trial] = Hamilt_phi(psi_trial, ibz, ispin);
                alpha = trial_step(dE1, E_trial - E0(n), params.line_search);
                b = 1;
                % Set threshold for first-order convergence criterion
                dE1_tol = abs(dE1) * params.convergence.dE1_scale;
                RMS_sum = RMS_sum + occ(n) * norm(R);
                E_last = E0(n);
            else
                % DIIS (i>1): build A/B with history 1:i, solve for b
                A = R_diis(:,1:i)' * s_diis(:,1:i);
                B = psi_diis(:,1:i)' * Spsi_diis(:,1:i);
                
                % Check Hermiticity of A and B matrices before symmetrizing
                A_hermitian_error = abs(A-A')./(A+A');
                B_hermitian_error = abs(B-B')./(B+B');
                
                A_err_max = max(max(A_hermitian_error));
                B_err_max = max(max(B_hermitian_error));
                
                if max(A_err_max, B_err_max) > 1e-10
                    error('DIIS: Non-Hermitian matrices detected before symmetrization: A_err=%.2e, B_err=%.2e', ...
                        A_err_max, B_err_max);
                end
                
                % Force matrices to be exactly Hermitian
                A = (A + A')/2;
                B = (B + B')/2;
                
                % Check eigenvalues of A matrix
                A_eigs = eig(A);
                A_min_eig = min(A_eigs);
                A_max_eig = max(A_eigs);
                % Check that B is positive definite and warn if condition number is too high
                [~, p] = chol(B);
                if p ~= 0
                    error('DIIS: Overlap matrix B is not positive definite at band %d, iter %d.', n, i);
                end
                % Calculate condition number of B
                B_eigs = eig(B);
                B_min_eig = min(B_eigs);
                B_max_eig = max(B_eigs);
                B_cond = B_max_eig / B_min_eig;
                
                if B_cond > 1e10
                    error('DIIS: Overlap matrix B has high condition number (%.2e) at band %d, iter %d. Numerical instability detected.', B_cond, n, i);
                end
                % Solve generalized eigenvalue problem
                [b, lambda] = eigs(A, B, 1, 'smallestreal');
                
                % Verify eigenvalues are real (exactly, not just small imaginary part)
                if ~isreal(lambda)
                    error('DIIS: Non-real eigenvalue detected: %.2e + %.2ei. This indicates a problem with Hermiticity of A or B.', real(lambda), imag(lambda));
                end
            end

            % Update using a single GEMV on (psi_diis - alpha*s_diis):
            % mathematically equal to (psi_diis*b) - alpha*(s_diis*b)
            psi = (psi_diis(:,1:i) - alpha * s_diis(:,1:i)) * b;
            % Let Hamilt_phi handle normalization as designed
            [Hpsi, Spsi, E_curr, psi] = Hamilt_phi(psi, ibz, ispin, true);
            
            % Check S-based norm after Hamilt_phi normalization (per-column form)
            S_norm = real(sum(conj(psi) .* Spsi, 1));
            psi_norm = norm(psi);
            
            % Check if S-based norm is negative (which indicates a problem with S)
            if S_norm < 0
                error('Negative S-norm detected after Hamilt_phi: %.2e. This suggests S matrix has negative eigenvalues.', S_norm);
            end

            % Per-band convergence checks (mirroring EDDRMM)
            dE = E_curr - E_last;
            E_last = E_curr;

                    if abs(dE) < params.convergence.dE_tol
                        break; % band converged by energy change
                    end

                    if i >= 2
                        if abs(dE1) < dE1_tol
                            break; % first-order criterion
                        end
                        if abs(occ(n)) < params.convergence.occ_min && abs(occ(n) * dE) < params.convergence.occ_min
                            break;
                        end
                    end
        end

        % Check if wavefunction is within reasonable bounds
        psi_norm = norm(psi);
        if ~isfinite(psi_norm) || psi_norm > 1e6
            error('Wavefunction norm too large: %.2e. Numerical instability detected.', psi_norm);
        end
        
        % Write back band state
        psi_ibz(:,n) = psi;
        E(n) = E_curr;
    end
end



% -----------------------------------------------------------------------------
% CHECK_CONVERGENCE - Check convergence and report progress
%
% conv_data = check_convergence(conv_data, params)
%
% Checks if the calculation has converged and reports progress.
%
% Inputs:
%   conv_data - Convergence data structure containing:
%               .iter - Current iteration number
%               .Etot - Current total energy
%               .Etot_prev - Previous total energy
%               .dE_sum - Weighted energy change sum
%               .RMS - Global RMS residual
%   params    - Algorithm parameters
%
% Outputs:
%   conv_data - Updated convergence data structure with:
%               .converged - Whether convergence has been achieved
%               .dEtot - Change in total energy
% -----------------------------------------------------------------------------
function conv_data = check_convergence(conv_data, params)
    % Calculate energy change
    conv_data.dEtot = conv_data.Etot - conv_data.Etot_prev;
    
    % Global logging
    if params.verbose && mod(conv_data.iter, params.log_every) == 0
        fprintf('=== ITERATION %d SUMMARY: RMS=%.3e Etot=%.8f dE=%.3e ===\n', ...
            conv_data.iter, conv_data.RMS, conv_data.Etot, conv_data.dEtot);
    end
    
    % Check for global convergence
    conv_data.converged = false;
    if conv_data.iter >= 2 && abs(conv_data.dEtot) < params.convergence.energy_tol && abs(conv_data.dE_sum) < params.convergence.energy_tol
        if params.verbose
            disp('=== GLOBAL CONVERGENCE ACHIEVED ===');
        end
        conv_data.converged = true;
    end
end

% -----------------------------------------------------------------------------
% SET_PARAMS - Set parameter with default if not provided
%
% value = set_params(params, field_name, default_value, description)
%
% Helper function to set a parameter value in a struct, with a default value
% if the field doesn't exist. Stores a description of the parameter.
%
% Inputs:
%   params        - Parameter struct
%   field_name    - Name of the field to get
%   default_value - Default value to use if field doesn't exist
%   description   - Description of the parameter
%
% Output:
%   value - The parameter value (either from params or the default)
% -----------------------------------------------------------------------------
function value = set_params(params, field_name, default_value)
    % Check if the field exists in the params struct
    if isfield(params, field_name)
        value = params.(field_name);
    else
        % Use the default value
        value = default_value;
        
        % Store the default in the params struct
        params.(field_name) = default_value;
    end
end


% -----------------------------------------------------------------------------
% TRIAL_STEP - Optimal step size calculation for line search
%
% alpha = trial_step(dE1, dE, params)
%
% Computes the optimal step size alpha for the wavefunction update using a
% quadratic model of the energy along the search direction.
%
% Inputs:
%   dE1    - First-order energy change estimate <R|s> (dot product of residual and search direction)
%   dE     - Energy difference E_trial - E0 from the trial step
%   params - Parameter struct containing alpha_min and alpha_max
%
% Output:
%   alpha  - Optimal step size, clamped to [alpha_min, alpha_max]
%
% The formula matches VASP's TRIAL_STEP subroutine:
%   A1 = -2*dE1
%   A2 = dE
%   alpha = -A1/(2*(A2-A1)) = dE1/(dE+2*dE1)
% -----------------------------------------------------------------------------
function alpha = trial_step(dE1, dE, line_search_params)
    denom = 2 * (dE + 2*dE1);
    if abs(denom) < 1e-12
        alpha = sign(2*dE1) * line_search_params.alpha_min;
    else
        alpha = 2*dE1 / denom;
    end
    alpha = sign(alpha) * min(max(abs(alpha), line_search_params.alpha_min), line_search_params.alpha_max);
end
