using LinearAlgebra, SparseArrays, BenchmarkTools, Plots

function pressure(sol::Vector,gamma)
    return (gamma-1.0)*(sol[4] - 0.5*(sol[2]^2+sol[3]^2)/sol[1])
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
        F4[:,i] = F[4,i]
    end
    return F
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



P(x,y) = sin(10pi*x) + cos(2pi*y)

h = 0.01
xrange = 0.0:h:1.0
N = length(xrange)

vals = [P(x,y) for y in xrange for x in xrange]

alpha = 0.48
A = filter_dy_matrix(N,alpha)
f = filter_dy_rhs(vals,N,alpha)
vy = A\f
A = filter_dx_matrix(N,alpha)
f = filter_dx_rhs(vals,N,alpha)
vx = A\f

println("Diff vx = ", norm(vals-vx))
println("Diff vy = ", norm(vals-vy))

# contour(xrange, xrange, reshape(vals,N,N), lw = 3)
# contour(xrange, xrange, reshape(vx,N,N), lw = 3)
