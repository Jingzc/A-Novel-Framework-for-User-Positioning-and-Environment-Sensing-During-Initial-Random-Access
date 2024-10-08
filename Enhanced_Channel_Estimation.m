function nmse = Enhanced_Channel_Estimation(y, A, prior, snr)
    % Received signal y
    % Perception matrix A
    % Prior to be restored X prior
    % Signal-to-noise ratio snr
    [~, SP, VP] = svd(A);
    h_gamp = image_by_EM_GAMP(y, A, snr);
    h_sbl = image_by_SBL(y, A);

    % OAMP parameters
    P = 0.01;
    u_g = 1;
    sigma = 0.5;
    v_g = sigma^2;
    iterNum = 30;

    [~, h_E_oamp] = E_OAMP(A, h, y, P, prior, v_n, u_g, v_g, iterNum);
    CC = ones(col_A, 1);
    [~, h_oamp, ~] = OAMP_SVD(A, VP, h, y, diag(SP), P, CC, v_n, u_g, v_g, iterNum, 1);
    
    nmse_h_gamp = (h_gamp - h)' * (h_gamp - h) / norm(h).^2;
    nmse_h_sbl = (h_sbl - h)' * (h_sbl - h) / norm(h).^2;
    nmse_h_oamp = (h_oamp - h)' * (h_oamp - h) / norm(h).^2;
    nmse_h_E_oamp = (h_E_oamp - h)' * (h_E_oamp - h) / norm(h).^2;
    nmse = [nmse_h_gamp nmse_h_sbl nmse_h_oamp nmse_h_E_oamp];
end
