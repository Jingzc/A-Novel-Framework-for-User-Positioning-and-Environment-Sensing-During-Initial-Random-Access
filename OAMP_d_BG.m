%% OAMP
function [MSE, x_hat] = E_OAMP(A, x, y, P, C, v_n, u_g, v_g, it)
    MSE = zeros(1, it);
    %Var = zeros(1, it);
    N = length(x);
    u_nle = P * C .* ones(N, 1);
    MSE0 = (u_nle - x)' * (u_nle - x) / N; 
    v_nle = P * (1 - P) * abs(C).^2 .* ones(N, 1);
    AHy = A' * y;
    AHA = A' * A;
    u_nle_p = zeros(N, 1);
    v_nle_p = zeros(N, 1);
    thres_0 = 1e-10;

    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE(AHA, AHy, u_nle, v_nle, v_n);
        v_le_p(v_le_p<=0) = mean(v_le_p); 
        [u_le, v_le] = Orth_d(u_le_p, v_le_p, u_nle, v_nle);
        % NLE
        u_le = real(u_le);
        v_le = v_le / 2;
        for k = 1 : N
            [u_nle_p(k), v_nle_p(k)] = Demodulation_BG(u_le(k), v_le(k), P,...
                abs(C(k)) * u_g, abs(C(k))^2 * v_g);
        end
        if mean(v_nle_p) <= thres_0
            tmp = (u_nle_p - x)' * (u_nle_p - x) / N;
            MSE(t:end) = max(tmp, thres_0);
            %Var(t:end) = thres_0;
            break
        end
        v_nle_p(v_nle_p <= 0) = mean(v_nle_p); 
        MSE(t) = (u_nle_p - x)' * (u_nle_p - x) / N;                
        %Var(t) = mean(v_nle_p);
        if t == it
            break
        end
        [u_nle, v_nle] = Orth_d(u_nle_p, v_nle_p, u_le, v_le);
    end
    MSE = [MSE0 MSE];
    x_hat = u_nle_p;
end

%% LE
function [u_post, v_post] = LE(AHA, AHy, u, v, v_n)
    V_post = (AHA / v_n + diag(1./v))^(-1);
    h = AHy / v_n + u ./ v;
    u_post = V_post * h;
    v_post = real(diag(V_post));
end

%% Orthogonalization in each dimension
function [u_orth, v_orth] = Orth_d(u_post, v_post, u_pri, v_pri)
    N = length(u_post);
    u_orth = zeros(N, 1);
    v_orth = zeros(N, 1);
    N = length(u_post);
    sp = 1 / (1 / mean(v_post) - 1 / mean(v_pri));
    for ii = 1 : N
        tmp = 1 / v_post(ii) - 1 / v_pri(ii);
        if tmp <= 0
            v_orth(ii) = sp;
        else
            v_orth(ii) = 1 / tmp;
        end
        u_orth(ii) = v_orth(ii) * (u_post(ii) / v_post(ii) - u_pri(ii) / v_pri(ii)); 
    end
end