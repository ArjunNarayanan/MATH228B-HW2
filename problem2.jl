using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot

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
    return k == N ? 1 : k+1
end

function front2(k::Int,N::Int)
    if k == N
        return 2
    elseif k == N-1
        return 1
    else
        return k+2
    end
end

function back(k::Int,N::Int)
    return k == 1 ? N : k-1
end

function back2(k::Int,N::Int)
    if k == 1
        return N-1
    elseif k == 2
        return N
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
    count = 1
    for j = 1:N
        for i = 1:N
            iplus = front(i,N)
            iminus = back(i,N)
            F = index_to_DOF(iplus,j,N)
            B = index_to_DOF(iminus,j,N)
            C = index_to_DOF(i,j,N)
            append!(I,[count,count,count])
            append!(J,[B,C,F])
            append!(vals,[1.0,4.0,1.0])
            count += 1
        end
    end
    return sparse(I,J,vals,N^2,N^2)
end

function pade_dy_matrix(N::Int)
    I = Int[]
    J = Int[]
    vals = Float64[]
    count = 1
    for j = 1:N
        jplus = front(j,N)
        jminus = back(j,N)
        for i = 1:N
            F = index_to_DOF(i,jplus,N)
            B = index_to_DOF(i,jminus,N)
            C = index_to_DOF(i,j,N)
            append!(I,[count,count,count])
            append!(J,[B,C,F])
            append!(vals,[1.0,4.0,1.0])
            count += 1
        end
    end
    return sparse(I,J,vals,N^2,N^2)
end

function compact_divergence(Fx::Vector,Fy::Vector,Ax::SparseMatrixCSC,Ay::SparseMatrixCSC,N::Int,h::Float64)
    rx = pade_dx_rhs(Fx,N,h)
    ry = pade_dy_rhs(Fy,N,h)
    return (Ax\rx + Ay\ry)
end

function euler_rhs(sol::Matrix,gamma::Float64,Ax::SparseMatrixCSC,Ay::SparseMatrixCSC,N::Int,h::Float64)
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
            append!(I,[count,count,count])
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
    count = 1
    for j = 1:N
        jplus = front(j,N)
        jminus = back(j,N)
        for i = 1:N
            F = index_to_DOF(i,jplus,N)
            B = index_to_DOF(i,jminus,N)
            C = index_to_DOF(i,j,N)
            append!(I,[count,count,count])
            append!(J,[B,C,F])
            append!(vals,[alpha,1.0,alpha])
            count += 1
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

function compact_filter(F::Vector,Rx::SparseMatrixCSC,Ry::SparseMatrixCSC,N::Int,alpha::Float64)
    rx = filter_dx_rhs(F,N,alpha)
    Fx = Rx\rx
    ry = filter_dy_rhs(Fx,N,alpha)
    Fy = Ry\ry
    return Fy
end

function filter_solution(sol::Matrix,Rx::SparseMatrixCSC,Ry::SparseMatrixCSC,N::Int,alpha::Float64)
    row1 = compact_filter(sol[1,:],Rx,Ry,N,alpha)
    row2 = compact_filter(sol[2,:],Rx,Ry,N,alpha)
    row3 = compact_filter(sol[3,:],Rx,Ry,N,alpha)
    row4 = compact_filter(sol[4,:],Rx,Ry,N,alpha)
    return vcat(row1',row2',row3',row4')
end

function stepRK4(sol::Matrix,gamma::Float64,Ax::SparseMatrixCSC,Ay::SparseMatrixCSC,N::Int,dx::Float64,dt::Float64)
    k1 = euler_rhs(sol,gamma,Ax,Ay,N,dx)
    k2 = euler_rhs(sol+0.5*dt*k1,gamma,Ax,Ay,N,dx)
    k3 = euler_rhs(sol+0.5*dt*k2,gamma,Ax,Ay,N,dx)
    k4 = euler_rhs(sol+dt*k3,gamma,Ax,Ay,N,dx)
    return sol+dt/6.0*(k1+k2+k3+k4)
end

function step_and_filter(sol::Matrix,gamma::Float64,Ax::SparseMatrixCSC,Ay::SparseMatrixCSC,
        Rx::SparseMatrixCSC,Ry::SparseMatrixCSC,N::Int,dx::Float64,dt::Float64,alpha::Float64)

    next_step = stepRK4(sol,gamma,Ax,Ay,N,dx,dt)
    filtered_next_step = filter_solution(next_step,Rx,Ry,N,alpha)
    return filtered_next_step
end

function run_steps(sol0,gamma,N,dx,dt,alpha,nsteps)
    Ax = pade_dx_matrix(N)
    Ay = pade_dy_matrix(N)
    Rx = filter_dx_matrix(N,alpha)
    Ry = filter_dy_matrix(N,alpha)

    sol = copy(sol0)
    for i = 1:nsteps
        sol = step_and_filter(sol,gamma,Ax,Ay,Rx,Ry,N,dx,dt,alpha)
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

function initial_velocity(uInf,vInf,r,x,y,xc,yc,b)
    u = uInf - b/(2pi)*exp(0.5*(1.0-r^2))*(y-yc)
    v = vInf + b/(2pi)*exp(0.5*(1.0-r^2))*(x-xc)
    return u,v
end

function initial_density(gamma,r,b)
    return (1.0 - (gamma - 1.0)*b^2/(8*gamma*pi^2)*exp(1.0-r^2))^(1.0/(gamma-1.0))
end

function initial_pressure(rho,gamma)
    return rho^gamma
end

function total_energy(p,rho,u,v,gamma)
    return 1.0/(gamma-1.0)*p + 0.5*rho .* (u.^2 + v.^2)
end

function initial_condition(xrange,gamma;b=0.5,xc=5.0,yc=5.0,uInf=0.1,vInf=0.0)
    N = length(xrange)
    ndofs = N^2
    density = zeros(ndofs)
    xVelocity = zeros(ndofs)
    yVelocity = zeros(ndofs)
    press = zeros(ndofs)

    count = 1
    for j in 1:N
        y = xrange[j]
        for i in 1:N
            x = xrange[i]
            r = sqrt((x-xc)^2 + (y-yc)^2)

            rho = initial_density(gamma,r,b)
            p = initial_pressure(rho,gamma)
            u,v = initial_velocity(uInf,vInf,r,x,y,xc,yc,b)

            density[count] = rho
            xVelocity[count] = u
            yVelocity[count] = v
            press[count] = p

            count += 1
        end
    end
    sol = zeros(4,ndofs)
    sol[1,:] = density
    sol[2,:] = density .* xVelocity
    sol[3,:] = density .* yVelocity
    sol[4,:] = total_energy(press,density,xVelocity,yVelocity,gamma)

    return sol
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

const gamma = 7.0/5.0
const alpha = 0.499
const final_time = 5*sqrt(2)
xc=5.0
yc=5.0
uInf=0.1
vInf=0.0
N = 32
Nx = N+1
dx = 10.0/N
dt,nsteps = time_step_size(final_time,dx)
xrange = range(0.0, stop = 10.0, length = Nx)

xT = xc + final_time*uInf
yT = yc

sol0 = initial_condition(xrange,gamma)
p0 = pressure(sol0,gamma)

Ax = pade_dx_matrix(Nx)
Ay = pade_dy_matrix(Nx)
Rx = filter_dx_matrix(Nx,alpha)
Ry = filter_dy_matrix(Nx,alpha)

sol = run_steps(sol0,gamma,Nx,dx,dt,alpha,nsteps)
p = pressure(sol,gamma)

xxs = [x for x in xrange for y in xrange]
yys = [y for x in xrange for y in xrange]
X = reshape(xxs,Nx,Nx)
Y = reshape(yys,Nx,Nx)
fig, ax = PyPlot.subplots()
ax.contour(X,Y,reshape(sol[1,:],Nx,Nx))
fig
