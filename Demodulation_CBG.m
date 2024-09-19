% Complex Bernoulli-Gaussian
function [u_post, v_post] = Demodulation_CBG(u, v, P, u_g, v_g)
    EXP_MAX = 50;
    EXP_MIN = -50;
    N = length(u);
    u_g = u_g * ones(N, 1);     % complex, not real
    % Post Bernoull
    alpha = (v + v_g) / v;
    beta = abs(u - u_g).^2 / (v + v_g) - abs(u).^2 / v;
    beta(beta > EXP_MAX) = EXP_MAX;
    beta(beta < EXP_MIN) = EXP_MIN;
    p_post = P ./ (P + (1-P) * alpha * exp(beta));
    % Post Gaussian
    v_pg = 1 / (1 / v + 1 / v_g);
    u_pg = v_pg * (u / v + u_g / v_g);
    % post u and v
    u_post = p_post .* u_pg;
    v_post = mean(((p_post - p_post.^2) .* (abs(u_pg).^2) + p_post * v_pg));
end