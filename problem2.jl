using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot, Statistics, Printf

function pressure(sol::Vector,gamma)
    return (gamma-1.0)*(sol[4] - 0.5*(sol[2]^2+sol[3]^2)/sol[1])
end

function pressure(sol::Matrix,gamma)
    ndofs = size(sol)[2]
    p = zeros(ndofs)
    for i in 1:ndofs
        p[i] = pressure(sol[:,i],gamma)
    end
    return p
end

function flux(sol::Vector,gamma)
    p = pressure(sol,gamma)
    ρu2 = sol[2]^2/sol[1]
    ρv2 = sol[3]^2/sol[1]
    ρuv = sol[2]*sol[3]/sol[1]
    u = sol[2]/sol[1]
    v = sol[3]/sol[1]
    ρE = sol[4]

    return [sol[2]         sol[3]
            ρu2+p          ρuv
            ρuv            ρv2+p
            u*(ρE+p)       v*(ρE+p)]
end

function flux(sol::Matrix,gamma)
    ndofs = size(sol)[2]
    F1 = zeros(2,ndofs)
    F2 = zeros(2,ndofs)
    F3 = zeros(2,ndofs)
    F4 = zeros(2,ndofs)
    for i = 1:ndofs
        F = flux(sol[:,i],gamma)
        F1[:,i] = F[1,:]
        F2[:,i] = F[2,:]
        F3[:,i] = F[3,:]
        F4[:,i] = F[4,:]
    end
    return F1,F2,F3,F4
end

function index_to_DOF(i::Int,j::Int,N::Int)
    return (j-1)*N+i
end

function front(k::Int,N::Int)
    return k == N ? 2 : k+1
end

function front2(k::Int,N::Int)
    if k == N
        return 3
    elseif k == N-1
        return 2
    else
        return k+2
    end
end

function back(k::Int,N::Int)
    return k == 1 ? N-1 : k-1
end

function back2(k::Int,N::Int)
    if k == 1
        return N-2
    elseif k == 2
        return N-1
    else
        return k-2
    end
end

function pade_dx_rhs(vals::Vector,i::Int,j::Int,N::Int,h::Float64)
    iplus = front(i,N)
    iminus = back(i,N)
    F = index_to_DOF(iplus,j,N)
    B = index_to_DOF(iminus,j,N)
    return 3.0*(vals[F] - vals[B])/h
end

function pade_dy_rhs(vals::Vector,i::Int,j::Int,N::Int,h::Float64)
    jplus = front(j,N)
    jminus = back(j,N)
    F = index_to_DOF(i,jplus,N)
    B = index_to_DOF(i,jminus,N)
    return 3.0*(vals[F] - vals[B])/h
end

function pade_dx_rhs(vals::Vector,N::Int,h::Float64)
    rhs = zeros(N^2)
    for j = 1:N
        for i = 1:N
            idx = index_to_DOF(i,j,N)
            rhs[idx] = pade_dx_rhs(vals,i,j,N,h)
        end
    end
    return rhs
end

function pade_dy_rhs(vals::Vector,N::Int,h::Float64)
    rhs = zeros(N^2)
    for j = 1:N
        for i = 1:N
            idx = index_to_DOF(i,j,N)
            rhs[idx] = pade_dy_rhs(vals,i,j,N,h)
        end
    end
    return rhs
end

function pade_dx_matrix(N::Int)
    I = Int[]
    J = Int[]
    vals = Float64[]
    for j = 1:N
        for i = 1:N
            iplus = front(i,N)
            iminus = back(i,N)
            F = index_to_DOF(iplus,j,N)
            B = index_to_DOF(iminus,j,N)
            C = index_to_DOF(i,j,N)
            append!(I,[C,C,C])
            append!(J,[B,C,F])
            append!(vals,[1.0,4.0,1.0])
        end
    end
    return sparse(I,J,vals,N^2,N^2)
end

function pade_dy_matrix(N::Int)
    I = Int[]
    J = Int[]
    vals = Float64[]
    for j = 1:N
        jplus = front(j,N)
        jminus = back(j,N)
        for i = 1:N
            F = index_to_DOF(i,jplus,N)
            B = index_to_DOF(i,jminus,N)
            C = index_to_DOF(i,j,N)
            append!(I,[C,C,C])
            append!(J,[B,C,F])
            append!(vals,[1.0,4.0,1.0])
        end
    end
    return sparse(I,J,vals,N^2,N^2)
end

function compact_divergence(Fx::Vector,Fy::Vector,Ax,Ay,N::Int,h::Float64)
    rx = pade_dx_rhs(Fx,N,h)
    ry = pade_dy_rhs(Fy,N,h)
    return (Ax\rx + Ay\ry)
end

function euler_rhs(sol::Matrix,gamma::Float64,Ax,Ay,N::Int,h::Float64)
    F1,F2,F3,F4 = flux(sol,gamma)
    DF1 = -1.0*compact_divergence(F1[1,:],F1[2,:],Ax,Ay,N,h)
    DF2 = -1.0*compact_divergence(F2[1,:],F2[2,:],Ax,Ay,N,h)
    DF3 = -1.0*compact_divergence(F3[1,:],F3[2,:],Ax,Ay,N,h)
    DF4 = -1.0*compact_divergence(F4[1,:],F4[2,:],Ax,Ay,N,h)
    return vcat(DF1',DF2',DF3',DF4')
end

function filter_dx_matrix(N::Int,alpha::Float64)
    I = Int[]
    J = Int[]
    vals = Float64[]
    count = 1
    for j = 1:N
        for i = 1:N
            iplus = front(i,N)
            iminus = back(i,N)
            F = index_to_DOF(iplus,j,N)
            B = index_to_DOF(iminus,j,N)
            C = index_to_DOF(i,j,N)
            append!(I,[C,C,C])
            append!(J,[B,C,F])
            append!(vals,[alpha,1.0,alpha])
            count += 1
        end
    end
    return sparse(I,J,vals,N^2,N^2)
end

function filter_dy_matrix(N::Int,alpha::Float64)
    I = Int[]
    J = Int[]
    vals = Float64[]
    for j = 1:N
        jplus = front(j,N)
        jminus = back(j,N)
        for i = 1:N
            F = index_to_DOF(i,jplus,N)
            B = index_to_DOF(i,jminus,N)
            C = index_to_DOF(i,j,N)
            append!(I,[C,C,C])
            append!(J,[B,C,F])
            append!(vals,[alpha,1.0,alpha])
        end
    end
    return sparse(I,J,vals,N^2,N^2)
end

function filter_coefficients(alpha::Float64)
    a = 5.0/8.0 + 3.0*alpha/4.0
    b = alpha + 0.5
    c = alpha/4.0 - 1.0/8.0
    return a,b,c
end

function filter_dx_rhs(vals::Vector,i::Int,j::Int,N::Int,a::Float64,b::Float64,c::Float64)
    iplus2 = front2(i,N)
    iminus2 = back2(i,N)
    iplus = front(i,N)
    iminus = back(i,N)
    F2 = index_to_DOF(iplus2,j,N)
    F = index_to_DOF(iplus,j,N)
    C = index_to_DOF(i,j,N)
    B = index_to_DOF(iminus,j,N)
    B2 = index_to_DOF(iminus2,j,N)
    return a*vals[C] + 0.5*c*(vals[F2] + vals[B2]) + 0.5*b*(vals[F]+vals[B])
end

function filter_dx_rhs(vals::Vector,N::Int,alpha::Float64)
    a,b,c = filter_coefficients(alpha)
    rhs = zeros(N^2)
    for j = 1:N
        for i = 1:N
            idx = index_to_DOF(i,j,N)
            rhs[idx] = filter_dx_rhs(vals,i,j,N,a,b,c)
        end
    end
    return rhs
end

function filter_dy_rhs(vals::Vector,i::Int,j::Int,N::Int,a::Float64,b::Float64,c::Float64)
    jplus2 = front2(j,N)
    jminus2 = back2(j,N)
    jplus = front(j,N)
    jminus = back(j,N)
    F2 = index_to_DOF(i,jplus2,N)
    F = index_to_DOF(i,jplus,N)
    C = index_to_DOF(i,j,N)
    B = index_to_DOF(i,jminus,N)
    B2 = index_to_DOF(i,jminus2,N)
    return a*vals[C] + 0.5*c*(vals[F2] + vals[B2]) + 0.5*b*(vals[F]+vals[B])
end

function filter_dy_rhs(vals::Vector,N::Int,alpha::Float64)
    a,b,c = filter_coefficients(alpha)
    rhs = zeros(N^2)
    for j = 1:N
        for i = 1:N
            idx = index_to_DOF(i,j,N)
            rhs[idx] = filter_dy_rhs(vals,i,j,N,a,b,c)
        end
    end
    return rhs
end

function compact_filter(F::Vector,Rx,Ry,N::Int,alpha::Float64)
    rx = filter_dx_rhs(F,N,alpha)
    Fx = Rx\rx
    ry = filter_dy_rhs(Fx,N,alpha)
    Fy = Ry\ry
    return Fy
end

function filter_solution(sol::Matrix,Rx,Ry,N::Int,alpha::Float64)
    row1 = compact_filter(sol[1,:],Rx,Ry,N,alpha)
    row2 = compact_filter(sol[2,:],Rx,Ry,N,alpha)
    row3 = compact_filter(sol[3,:],Rx,Ry,N,alpha)
    row4 = compact_filter(sol[4,:],Rx,Ry,N,alpha)
    return vcat(row1',row2',row3',row4')
end

function stepRK4_with_filter(sol::Matrix,gamma::Float64,Ax,Ay,
        Rx,Ry,N::Int,dx::Float64,dt::Float64,alpha::Float64)

    k1 = euler_rhs(sol,gamma,Ax,Ay,N,dx)
    k1 = filter_solution(k1,Rx,Ry,N,alpha)

    k2 = euler_rhs(sol+0.5*dt*k1,gamma,Ax,Ay,N,dx)
    k2 = filter_solution(k2,Rx,Ry,N,alpha)

    k3 = euler_rhs(sol+0.5*dt*k2,gamma,Ax,Ay,N,dx)
    k3 = filter_solution(k3,Rx,Ry,N,alpha)

    k4 = euler_rhs(sol+dt*k3,gamma,Ax,Ay,N,dx)
    k4 = filter_solution(k1,Rx,Ry,N,alpha)

    next_step = sol+dt/6.0*(k1+2.0*k2+2.0*k3+k4)
    next_step = filter_solution(next_step,Rx,Ry,N,alpha)

    return next_step
end

function run_steps(sol0,gamma,N,dx,dt,alpha,nsteps)
    Ax = lu(pade_dx_matrix(N))
    Ay = lu(pade_dy_matrix(N))
    Rx = lu(filter_dx_matrix(N,alpha))
    Ry = lu(filter_dy_matrix(N,alpha))

    sol = copy(sol0)
    for i = 1:nsteps
        sol = stepRK4_with_filter(sol,gamma,Ax,Ay,Rx,Ry,N,dx,dt,alpha)
    end
    return sol
end

function time_step_size(final_time::Float64,dx::Float64;step_factor=0.3)
    nsteps = 2
    dt = final_time/nsteps
    while dt > step_factor*dx
        nsteps *= 2
        dt = final_time/nsteps
    end
    return dt, nsteps
end

function vortex_initial_velocity(uInf,vInf,r,x,y,xc,yc,b)
    u = uInf - b/(2pi)*exp(0.5*(1.0-r^2))*(y-yc)
    v = vInf + b/(2pi)*exp(0.5*(1.0-r^2))*(x-xc)
    return u,v
end

function vortex_initial_density(gamma,r,b)
    return (1.0 - (gamma - 1.0)*b^2/(8*gamma*pi^2)*exp(1.0-r^2))^(1.0/(gamma-1.0))
end

function kelvin_helmholtz_initial_density(x,y)
    if abs(y - 0.5) < 0.15 + sin(2*pi*x)/200.0
        return 2.0
    else
        return 1.0
    end
end

function vortex_initial_pressure(rho,gamma)
    return rho^gamma
end

function total_energy_into_density(p,rho,u,v,gamma)
    return 1.0/(gamma-1.0)*p + 0.5*rho .* (u.^2 + v.^2)
end

function vortex_initial_condition(xrange,gamma,xc,yc,uInf,vInf;b=0.5)
    N = length(xrange)
    ndofs = N^2
    density = zeros(ndofs)
    xVelocity = zeros(ndofs)
    yVelocity = zeros(ndofs)
    press = zeros(ndofs)

    for j in 1:N
        y = xrange[j]
        for i in 1:N
            idx = index_to_DOF(i,j,N)

            x = xrange[i]
            r = sqrt((x-xc)^2 + (y-yc)^2)

            rho = vortex_initial_density(gamma,r,b)
            p = vortex_initial_pressure(rho,gamma)
            u,v = vortex_initial_velocity(uInf,vInf,r,x,y,xc,yc,b)

            density[idx] = rho
            xVelocity[idx] = u
            yVelocity[idx] = v
            press[idx] = p
        end
    end
    sol = zeros(4,ndofs)
    sol[1,:] = density
    sol[2,:] = density .* xVelocity
    sol[3,:] = density .* yVelocity
    sol[4,:] = total_energy_into_density(press,density,xVelocity,yVelocity,gamma)

    return sol
end

function kelvin_helmholtz_initial_condition(xrange)
    N = length(xrange)
    ndofs = N^2
    density = zeros(ndofs)
    for j in 1:N
        y = xrange[j]
        for i in 1:N
            x = xrange[i]
            idx = index_to_DOF(i,j,N)
            density[idx] = kelvin_helmholtz_initial_density(x,y)
        end
    end

    xVelocity = density .- 1.0
    yVelocity = zeros(ndofs)
    press = 3.0*ones(ndofs)

    sol = zeros(4,ndofs)
    sol[1,:] = density
    sol[2,:] = density .* xVelocity
    sol[3,:] = density .* yVelocity
    sol[4,:] = total_energy_into_density(press,density,xVelocity,yVelocity,gamma)

    return sol
end

function solution_error_infinity_norm(sol::Matrix,exact::Matrix)
    difference = sol - exact
    e1 = maximum(abs.(difference[1,:]))
    e2 = maximum(abs.(difference[2,:]))
    e3 = maximum(abs.(difference[3,:]))
    e4 = maximum(abs.(difference[4,:]))
    return [e1,e2,e3,e4]
end

function solution_error_L2_norm(sol::Matrix,exact::Matrix)
    difference = sol - exact
    e1 = norm(difference[1,:])
    e2 = norm(difference[2,:])
    e3 = norm(difference[3,:])
    e4 = norm(difference[4,:])
    return [e1,e2,e3,e4]
end

function plot_velocity_field(sol,Nx,xrange)
    xxs = [x for x in xrange for y in xrange]
    yys = [y for x in xrange for y in xrange]
    fig, ax = PyPlot.subplots()
    U = reshape(sol[2,:], Nx, Nx)
    V = reshape(sol[3,:], Nx, Nx)
    ax.quiver(xxs,yys,U,V)
    # ax.scatter([xT],[yT],s=50,color="r")
    return fig
end

function compact_pade(vals,Nx,dx)
    Ax = pade_dx_matrix(Nx)
    Ay = pade_dy_matrix(Nx)
    rx = pade_dx_rhs(vals,Nx,dx)
    ry = pade_dy_rhs(vals,Nx,dx)
    return (Ax\rx, Ay\ry)
end

function pade_error(Nrange::Vector,grid_size,P,dPx,dPy)
    Nxrange = Nrange .+ 1
    dx = grid_size ./ Nrange
    ex = zeros(length(Nxrange))
    ey = zeros(length(Nxrange))
    for (idx,Nx) in enumerate(Nxrange)
        xrange = range(0.0, stop = 1.0, length = Nx)
        vals = [P(x,y) for y in xrange for x in xrange]
        dvx = [dPx(x,y) for y in xrange for x in xrange]
        dvy = [dPy(x,y) for y in xrange for x in xrange]
        vx, vy = compact_pade(vals,Nx,dx[idx])
        ex[idx] = maximum(abs.(vx - dvx))
        ey[idx] = maximum(abs.(vy - dvy))
    end
    return ex, ey, dx
end

function mean_convergence_rate(err,dx)
    rate = mean(diff(log.(err)) ./ diff(log.(dx)))
    return rate
end

function test_pade_convergence()
    P(x,y) = cos(2pi*x) + sin(2pi*y)
    dPx(x,y) = -2pi*sin(2pi*x)
    dPy(x,y) = 2pi*cos(2pi*y)
    ex, ey, dx = pade_error([16,32,64,128], 1.0, P, dPx, dPy)
    cx = mean_convergence_rate(ex, dx)
    cy = mean_convergence_rate(ey, dx)
    return cx, cy
end

function vortex_convergence_rate(Srange,alpha)
    gamma = 7.0/5.0
    final_time = 5*sqrt(2)
    xc=5.0
    yc=5.0
    uInf=0.1
    vInf=0.0
    xT = xc + final_time*uInf
    yT = yc
    dxrange = 10.0 ./ Srange
    err_range = zeros(4,length(Srange))
    for (idx,S) in enumerate(Srange)
        dx = dxrange[idx]
        dt,nsteps = time_step_size(final_time,dx)
        N = S+1
        xrange = range(0.0, stop = 10.0, length = N)
        sol0 = vortex_initial_condition(xrange,gamma,xc,yc,uInf,vInf)
        sol = run_steps(sol0,gamma,N,dx,dt,alpha,nsteps)
        exact = vortex_initial_condition(xrange,gamma,xT,yT,uInf,vInf)
        err = solution_error_infinity_norm(sol,exact)
        err_range[:,idx] = err
    end
    return err_range, dxrange
end

function subplot_convergence(ax,dx,err,title)
    ax.loglog(dx,err,"-o",linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(L"Step size $h$")
    ax.set_ylabel(L"L_\infty error")
    rate = mean_convergence_rate(err,dx)
    annotation = @sprintf "mean slope = %1.1f" rate
    ax.annotate(annotation, (0.5, 0.2), xycoords = "axes fraction")
    ax.grid()
end

function plot_convergence(err,dx;figsize=(8,8),filename = "")
    ρ = err[1,:]
    ρu = err[2,:]
    ρv = err[3,:]
    ρE = err[4,:]
    fig, ax = PyPlot.subplots(2,2,figsize=figsize)
    subplot_convergence(ax[1,1],dx,ρ,L"Convergence of $\rho$")
    subplot_convergence(ax[1,2],dx,ρE,L"Convergence of $\rho E$")
    subplot_convergence(ax[2,1],dx,ρu,L"Convergence of $\rho u$")
    subplot_convergence(ax[2,2],dx,ρv,L"Convergence of $\rho v$")
    fig.tight_layout()
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

function plot_density(rho,xrange)
    N = length(xrange)
    xxs = reshape([x for x in xrange for y in xrange], N, N)
    yys = reshape([y for x in xrange for y in xrange], N, N)
    fig, ax = PyPlot.subplots()
    cbar = ax.contourf(xxs,yys,reshape(rho,N,N)')
    fig.colorbar(cbar)
    return fig
end

gamma = 7.0/5.0

# final_time = 5*sqrt(2)
final_time = 1.0

xc=5.0
yc=5.0
uInf=0.1
vInf=0.0
xT = xc + final_time*uInf
yT = yc
S = 128
alpha = 0.48

dx = 10.0 / S
dt,nsteps = time_step_size(final_time,dx)
N = S+1

# xrange = range(0.0, stop = 10.0, length = N)
# sol0 = vortex_initial_condition(xrange,gamma,xc,yc,uInf,vInf)

# xrange = range(0.0, stop = 1.0, length = N)
# sol0 = kelvin_helmholtz_initial_condition(xrange)
# #
# sol = run_steps(sol0,gamma,N,dx,dt,alpha,10)
# #
# plot_density(sol[1,:],xrange)
Srange = [16,32,64]
err, dx = vortex_convergence_rate(Srange,alpha)
fig = plot_convergence(err,dx)
