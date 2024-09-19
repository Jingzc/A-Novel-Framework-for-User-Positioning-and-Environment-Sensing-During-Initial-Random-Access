%% OAMP (SVD)
function [MSE, x_hat, T] = OAMP_SVD(A, V, x, y, dia, P, C, v_n, u_g, v_g, it, flag)
    MSE = zeros(1, it);
    %Var = zeros(1, it);
    M = length(y);
    N = length(x);
    u_nle = mean(C) * P * ones(N, 1);
    MSE0 = (u_nle - x)' * (u_nle - x) / N;  
    v_nle = mean(abs(C).^2) * P * (1 - P);
    AHy = A' * y;
    thres_0 = 1e-10;
    u_nle_p = zeros(N, 1);
    vtmp = zeros(N, 1);

    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE(u_nle, v_nle, V, AHy, dia, v_n, M, N);
        [u_le, v_le] = Orth(u_le_p, v_le_p, u_nle, v_nle);
        tt = real(u_le);
        % NLE
        for k = 1 : N
            %[u_nle_p(k), vtmp(k)] = Demodulation(u_le(k), v_le, P, C(k));
            [u_nle_p(k), vtmp(k)] = Demodulation_BG(u_le(k), v_le, P, u_g, v_g);
        end
        v_nle_p = mean(vtmp);
        if v_nle_p <= thres_0
            tmp = (u_nle_p - x)' * (u_nle_p - x) / N;
            MSE(t:end) = max(tmp, thres_0);
            %Var(t:end) = thres_0;
            T = t;
            break
        end
        MSE(t) = (u_nle_p - x)' * (u_nle_p - x) / N;                % MSE
        %Var(t) = v_nle_p;
        if t == it
            T = t;
            break
        % elseif部分是判断如果MSE开始发散，就停止算法
        elseif flag == 1 && t > 1
            if MSE(t) > MSE(t-1)
                T = t;
                break
            end
        end
        [u_nle, v_nle] = Orth(u_nle_p, v_nle_p, u_le, v_le);
    end
    MSE = [MSE0 MSE];
    x_hat = u_nle_p;
end

%% LE for OAMP (SVD)
function [u_post, v_post] = LE(u, v, V, AHy, dia, v_n, M, N)
    rho = v_n / v;
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    u_post = V * (D .* (V' * (AHy + rho * u)));
    v_post = v_n / N * sum(D);
end

%% Orthogonalization
function [u_orth, v_orth] = Orth(u_post, v_post, u_pri, v_pri)
    v_orth = 1 / (1 / v_post - 1 / v_pri);
    u_orth = v_orth * (u_post / v_post - u_pri / v_pri);  
end
