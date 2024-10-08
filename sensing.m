function res = sensing(x_LS, y_LS, x_bs, y_bs, d_i, phi_i, phi_beam)
    % Separate the reflection path and the scattered paths
    d_reflect = d_i(1);
    d_i = d_i(2 : end);
    phi_reflect = phi_i(1);


    % Constructing the reflection ellipse
    d_0 = sqrt((x_bs - x_LS)^2 + (y_bs - y_LS)^2);
    theta = asin((y_LS - y_bs) / d_0);
    theta = theta + 2 * pi;
    a = d_reflect / 2;
    b = sqrt((d_reflect / 2)^2 - (d_0 / 2)^2);
    
    x_u = x_LS; y_u = y_LS;
    x_c = (x_bs + x_u) / 2;
    y_c = (y_bs + y_u) / 2;
    
    A = cos(theta)^2 / a^2 + sin(theta)^2 / b^2;
    B = 2 * sin(theta) * cos(theta) * (1 / a^2 - 1 / b^2);
    C = sin(theta)^2 / a^2 + cos(theta)^2 / b^2;

    syms r;
    equ = (A * (x_bs + r * cos(phi_reflect) - x_c)^2 + B*(x_bs + r * cos(phi_reflect) - x_c)*(y_bs + r * sin(phi_reflect) - y_c) ...
        + C * (y_bs + r * sin(phi_reflect) - y_c)^2 == 1);
    answ = double(solve(equ));
    
    % Reflection point coordinates
    x_0 = x_bs + answ(2) * cos(phi_reflect);
    y_0 = y_bs + answ(2) * sin(phi_reflect);
    
    % The straight line where the reflecting surface lies
    k = x_0 * (2 * A * (x_0 - x_c) + B * (y_0 - y_c));
    m = y_0 * (2 * C * (y_0 - y_c) + B * (x_0 - x_c));
    n = -(2 * A * (x_0 - x_c) + B * (y_0 - y_c)) * x_0^2 - (2 * C * (y_0 - y_c) + B * (x_0 - x_c)) * y_0^2;
    K = - k / m;
    M = - n / m;
    res = [];
    for i = 1 : length(d_i)
        % Find the intersection point of the reflecting surface and the scattering ellipse
        a = d_i(i) / 2;
        b = sqrt((d_i(i) / 2)^2 - (d_0 / 2)^2);
        A = cos(theta)^2 / a^2 + sin(theta)^2 / b^2;
        B = 2 * sin(theta) * cos(theta) * (1 / a^2 - 1 / b^2);
        C = sin(theta)^2 / a^2 + cos(theta)^2 / b^2;
    
        a_eq = A + B * K + C * K^2;
        b_eq = B * M + 2 * C * K * M - 2 * A * x_c - B * y_c - 2 * C * K * y_c - B * K * x_c;
        c_eq = C * M^2 - 2 * C * M * y_c - B * M * x_c + A * x_c^2 + B * x_c * y_c + C * y_c^2 - 1;
        x = [(-b_eq + sqrt(b_eq^2 - 4 * a_eq * c_eq)) / (2 * a_eq), (-b_eq - sqrt(b_eq^2 - 4 * a_eq * c_eq)) / (2 * a_eq)];
        y = (-n - k * x) / m;
        pos = [x.' y.'];

        % Check whether the scattering point is within the beam coverage
        for l = 1 : size(pos, 1)
            v = [pos(l, 1) - x_bs pos(l, 2) - y_bs];
            dir_t = [cos(phi_reflect) sin(phi_reflect)];
            t1 = sum(v.*dir_t) / norm(v);
            t2 = 2 / (N_t * cos(phi_beam));
            if  t1 >= cos(t2) 
                res = [res; pos(l, :)];
            end
        end
    end
end
