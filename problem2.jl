using LinearAlgebra, SparseArrays

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

function index_to_DOF(i::Int,j::Int,N::Int)
    return (j-1)*N+i
end

function front(k::Int,N::Int)
    return k == N ? 1 : k+1
end

function back(k::Int,N::Int)
    return k == 1 ? N : k-1
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

P(x,y) = sin(pi*x) + cos(pi*y)
dPx(x,y) = pi*cos(pi*x)
dPy(x,y) = -pi*sin(pi*y)

h = 0.5
xrange = 0.0:h:1.0
N = length(xrange)

vals = [P(x,y) for y in xrange for x in xrange]
dvals = [dPy(x,y) for y in xrange for x in xrange]
dvals = reshape(dvals,N,N)

A = pade_dy_matrix(N)
f = pade_dy_rhs(vals,N,h)
df = A\f
df = reshape(df,N,N)
