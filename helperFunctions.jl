using LinearAlgebra
using Distributions
using Random
using Statistics

function generateSubspace(ambientSpace::Int, latentSpace::Int; seed::Int=rand(1:100000))
    return svd(rand(Xoshiro(seed),Normal(0,1),ambientSpace,latentSpace)).U
end

function generateData(U::Matrix, v::Vector = ones(2), points::Vector = 10*ones(Int,2); coordinateWindow::Real = 10, coordinateType::Symbol = :uniform, seed::Int=rand(1:100000))
    ambientSpace, dimSubspace = size(U)
    totalPoints = sum(points)
    if coordinateType === :uniform
        X = U*rand(Xoshiro(seed),Uniform(-coordinateWindow,coordinateWindow),dimSubspace,totalPoints)
    end
    if coordinateType === :gaussian
        X = U*rand(Xoshiro(seed), Normal(0, (1/sqrt(3))*coordinateWindow ), dimSubspace, totalPoints)
    end
    Y = zeros(ambientSpace,totalPoints)
    Y[:,1:points[1]] = X[:,1:points[1]] +  rand(Xoshiro(seed),Normal(0,sqrt(v[1])), ambientSpace, points[1])
    Y[:,(points[1]+1):end] = X[:,(points[1]+1):end] +  rand(Xoshiro(seed),Normal(0,sqrt(v[2])), ambientSpace, points[2])
    return Y
end

function generateDataHomo(U::Matrix, v::Real = 0.1, totalPoints::Int = 6; coordinateWindow::Real = 10, coordinateType::Symbol = :uniform, seed::Int=rand(1:100000))
    ambientSpace, dimSubspace = size(U)
    if coordinateType === :uniform
        X = U*rand(Xoshiro(seed),Uniform(-coordinateWindow,coordinateWindow),dimSubspace,totalPoints)
    end
    if coordinateType === :gaussian
        X = U*rand(Xoshiro(seed), Normal(0, (1/sqrt(3))*coordinateWindow ), dimSubspace, totalPoints)
    end
    Y = X +  rand(Xoshiro(seed),Normal(0,sqrt(v)), ambientSpace, totalPoints)
    return Y
end

function affinityError(Ut::Matrix, U::Matrix)
    return norm(U*U' - Ut*Ut', 2)/norm(Ut*Ut', 2)
end

