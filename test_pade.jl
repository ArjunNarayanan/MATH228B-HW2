using SparseArrays, LinearAlgebra, PyPlot, Statistics

f(x) = sin(2pi*x)
df(x) = 2pi*cos(2pi*x)

function pade_rhs(vals,h)
    N = length(vals)
    rhs = zeros(N)
    rhs[1] = vals[2] - vals[N-1]
    for i = 2:N-1
        rhs[i] = vals[i+1] - vals[i-1]
    end
    rhs[N] = vals[2] - vals[N-1]
    return 3.0/h*rhs
end

function compact_pade(vals)
    Nx = length(vals)
    N = Nx-1
    h = 1.0/N
    A = spdiagm(0 => 4*ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1))
    A[1,Nx-1] = 1.0
    A[Nx,2] = 1.0
    rhs = pade_rhs(vals,h)
    return A\rhs
end

N = [16,32,64,128]
h = 1.0 ./ N
Nx = N .+ 1

xrange = [range(0.0, stop = 1.0, length = n) for n in Nx]

vals = [f.(xr) for xr in xrange]
dvals = [df.(xr) for xr in xrange]
dv = [compact_pade(v) for v in vals]

err = [maximum(abs.(dv[i] - dvals[i])) for i = 1:length(N)]

fig, ax = PyPlot.subplots()
ax.loglog(h,err,"-o")
ax.grid()
fig
slope = mean(diff(log.(err)) ./ diff(log.(h)))
