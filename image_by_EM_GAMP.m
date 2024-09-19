function xhat = image_by_EM_GAMP(y, A, SNR_dB)
    addpath('.\gampmatlab20210328\latest\trunk\code\main');
    addpath('.\gampmatlab20210328\latest\trunk\code\stateEvo');
    addpath('.\gampmatlab20210328\latest\trunk\code\EMGMAMPnew');
    addpath('.\gampmatlab20210328\latest\trunk\code\turboGAMP\Functions');
    addpath('.\gampmatlab20210328\latest\trunk\code\HUTAMP');
    addpath('.\gampmatlab20210328\latest\trunk\code\neural\connectivity');
    addpath('.\gampmatlab20210328\latest\trunk\code\VAMP');
    T = 20;
    tol = 1e-4;
    
    
    clear optEM optGAMP;
    optEM.heavy_tailed = false;
    optEM.robust_gamp  = true;
    optEM.SNRdB = SNR_dB;
%   optEM.sig_dim = 'joint';
    
    optGAMP.nit = T;   % number of iterations
    optGAMP.tol = tol; % convergence tolerance
    optGAMP.removeMean = true;
    optGAMP.prt=1;
    [xhat, ~, ~, ~, ~] = EMGMAMP(y, A, optEM, optGAMP);
    
    %[xhat, ~] = VampSlmEst(x, y, A,optGAMP);
    %[xhat, ~, ~, ~] = EMturboGAMP(y, A, optEM, optGAMP);
    %[xhat, ~, ~] = HUTAMP(y, A, optEM, optGAMP);
    
end