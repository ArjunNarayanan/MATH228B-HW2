r_exact = zeros(N+1,N+1); u_exact = zeros(N+1,N+1);
    v_exact = zeros(N+1,N+1); p_exact = zeros(N+1,N+1);
    r_inf = 1; u_inf = 0.1; v_inf = 0; p_inf = 1;
    x_exact = x_c + FinalTime*u_inf;
    
    for i = 1:N+1
        for j = 1:N+1
            d = (x(j)-x_exact)^2 + (y(i)-y_c)^2;
            r_exact(i,j) = (1- (gamma-1)*b^2/(8*gamma*pi^2)*exp(1-d))^(1/(gamma-1));
            u_exact(i,j) = u_inf - b/2/pi*exp((1-d)/2)*(y(i)-y_c);
            v_exact(i,j) = v_inf + b/2/pi*exp((1-d)/2)*(x(j)-x_exact);
            p_exact(i,j) = r_exact(i,j)^gamma;
        end
    end
    
    ru_exact = r_exact.*u_exact; rv_exact = r_exact.*v_exact; 
    rE_exact = p_exact/(gamma-1) + 1/2*r_exact.*(u_exact.^2+v_exact.^2);