function [x_LS, y_LS] = positioning(Path_LoS, Path_NLoS)  
% Path_LoS and Path_NLoS are the path information of the user channel, including distance and angle
    N1 = length(Path_LoS); N2 = length(Path_NLoS); 
    k_ij = []; b_ij = [];
    for k = 1 : N2
        d = Path_NLoS{k}(1);
        phi = Path_NLoS{k}(2);
        theta = Path_NLoS{k}(3);
        k_t = (sin(phi) + sin(theta)) / (cos(phi) + cos(theta));
        b_t = -k_t * (x_bs - d * cos(theta)) + y_bs - d * sin(theta);
        k_ij = [k_ij k_t];
        b_ij = [b_ij b_t];
    end
    x_LoS = 0; y_LoS = 0;
    if N1 ~= 0
        N1 = 1;
        x_LoS = x_bs + Path_LoS{1}(1) * cos(Path_LoS{1}(2));
        y_LoS = y_bs + Path_LoS{1}(1) * sin(Path_LoS{1}(2));
    end
    G = -(sum(k_ij))^2 - (N1 - N2) * (N1 + sum(k_ij.^2));
    E = -(sum(k_ij)) * (y_LoS - sum(b_ij)) - (N1 - N2) * (x_LoS - sum(k_ij.*b_ij));
    F = sum(k_ij) * (x_LoS - sum(k_ij.*b_ij)) - (N1 + N3 + sum(k_ij.^2)) * (y_LoS - sum(b_ij));
    x_LS = E / G;
    y_LS = F / G;
end