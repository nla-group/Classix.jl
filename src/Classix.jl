module Classix

using LinearAlgebra: norm, eigen, Symmetric, dot 
using Statistics: mean, median
using Arpack: svds
using SparseArrays: spzeros

export classix

function classix(data::Matrix{Float64}; radius::Float64=0.2, minPts::Int=1, merge_tiny_groups::Bool=true)
    # CLASSIX - Fast and explainable clustering based on sorting.
    #
    # inputs   * data - a matrix in which each row is a data point
    #          * radius - (hyperparameter) gives a scale to the desired clusters
    #          * minPts - (hyperparameter) minimum number of points in a cluster
    #          * merge_tiny_groups - boolean
    #
    # returns  * cluster labels of the data
    #          * function to explain the clustering
    #          * out - a named tuple with fields
    #                .cs    -  cluster size (#points in each cluster)
    #                .dist  -  #distance computations during aggregation
    #                .gc    -  group center indices 
    #                .scl   -  data scaling parameter
    #                .t1... -  timings of CLASSIX's phases (in seconds)
    #
    # This is a Julia implementation of the CLASSIX clustering algorithm:
    #   X. Chen & S. Güttel. Fast and explainable clustering based on sorting. 
    #   Technical Report arXiv:2202.01456, arXiv, 2022. 
    #   https://arxiv.org/abs/2202.01456
    #
    
    tic = time()
    x, u, ind, half_r2, half_nrm2, scl, U = prepare(data, radius)
    toc = time()
    t1_prepare = toc-tic

    tic = time()
    label, gc, gs, dist, group_label = aggregate(x, u, half_r2, half_nrm2, radius)
    toc = time()
    t2_aggregate = toc - tic

    tic = time()
    cs, gc_label, gc_half_nrm2 = merge_groups(x, label, gc, gs, half_nrm2, radius, minPts, merge_tiny_groups)
    toc = time()
    t3_merge = toc - tic
    
    # At this point we have consecutive cluster gc_label (values 1,2,3,...) for each group center, 
    # and cs contains the total number of points for each cluster label.
    tic = time()
    min_pts!(label, gc, gs, cs, gc_label, gc_half_nrm2, ind, group_label, minPts)
    toc = time()
    t4_minPts = toc - tic
    
    out = (;cs, dist, gc, scl, t1_prepare, t2_aggregate, t3_merge, t4_minPts)

    explain = explain_fun(x, label, group_label, U, out, radius, minPts) 

    return label, explain, out
end

function prepare(data::Matrix{Float64}, radius::Float64)
    size(data,1) < size(data,2) && warn("Fewer data points than features. Check that each row corresponds to a data point.")
    size(data,2) > 5000 && warn("More than 5000 features. Consider applying some dimension reduction first.")
    
    x = collect(data')  # transpose. much faster when data points are stored column-wise
    x .-= mean(x, dims=2)
    scl = median(norm.(eachcol(x)))
    scl == 0.0 && (scl = 1.0) # prevent zero division
    x ./= scl
    
    U = similar(data, size(x,2), 2)
    if size(x,1) > 5000    # SVDS / SVD
        S = similar(data,2)
        U, S .= svds(x', nsv=2)
        U .*= S'
    else    # PCA via eigenvalues (faster & we don't need high accuracy)
        if size(x,1)==1
            U[:,1] .= x'
            U[:,2] .= 0
        else
            xtx = Symmetric(x*x')
            d,V = eigen(xtx)
            i = sortperm(d, by=abs, rev=true)
            U .= x'*V[:,i[1:2]]
        end
    end
    
    U[:,1] .*= sign(-U[1,1]) # flip to enforce deterministic output
    U[:,2] .*= sign(-U[1,2]) # also for plotting
    u = U[:,1]                    # scores
    ind = sortperm(u)
    u .= u[ind]
    x = x[:,ind]
    half_r2 = radius^2/2
    half_nrm2 = sum(x.^2,dims=1)./2   # ,1 needed for 1-dim feature

    return x, u, ind, half_r2, half_nrm2, scl, U
end

function aggregate(x::Matrix{Float64}, u::Vector{Float64}, half_r2::Float64, half_nrm2::Matrix{Float64}, radius::Float64)
    n = size(x,2)
    label = zeros(Int, n)
    lab = 1
    dist = 0    # no. distances comput.
    gc = Int[]     # indices of group centers (in sorted array)
    gs = Int[]     # group size
    for i = 1:n
        label[i] > 0 && continue
        label[i] = lab
        push!(gc,i)
        push!(gs,1)
        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.
    
        for j = i+1:n
            label[j] > 0 && continue
            u[j] - u[i] > radius && break # early termination (uj - ui > radius)
            dist += 1
            ip = @views dot(x[:,i],x[:,j])
            if half_nrm2[j] - ip <= rhs   # if vecnorm(xi-xj) <= radius
                label[j] = lab
                gs[end] += 1
            end
        end
        lab += 1
    end
    group_label = copy(label) # store original group labels

    return label, gc, gs, dist, group_label
end

function merge_groups(x::Matrix{Float64}, label::Vector{Int}, gc::Vector{Int}, gs::Vector{Int}, half_nrm2::Matrix{Float64}, radius::Float64, minPts::Int, merge_tiny_groups::Bool)
    gc_x = x[:,gc]
    gc_label = label[gc]  # will be [1,2,3,...]
    gc_half_nrm2 = half_nrm2[gc]
    A = spzeros(Bool, length(gc),length(gc)) # adjacency of group centers
    
    for i ∈ eachindex(gc)
        if !merge_tiny_groups && gs[i] < minPts # tiny groups cannot take over large ones
            continue
        end
    
        xi = view(gc_x,:,i)      # current group center coordinate
        rhs = (1.5*radius)^2/2 - gc_half_nrm2[i]  # rhs of norm ineq.
    
        # get id = (norm.(eachcol(xi - gc_x)) ≤ 1.5*radius); and igore id's < i
        id = ((gc_half_nrm2 .- @views gc_x'*xi) .≤ rhs)
        id[1:i-1] .= 0

        !merge_tiny_groups && (id .&= (gs .≥ minPts)) # tiny groups are not merged into larger ones
      
        A[id,i] .= 1 # adjacency, keep track of merged groups 
    
        gcl = unique(gc_label[id]) # get all the affected group center labels
        # TODO: could speedup unique by exploiting sorting?
    
        minlab = minimum(gcl)
        for L ∈ gcl
            gc_label[gc_label .== L] .= minlab  # important: need to relabel all of them,
        end                                     # not just the ones in id, as otherwise
                                                # groups that joined out of
                                                # order might stay disconnected
    end

    # rename labels to be 1,2,3,... and determine cluster sizes
    ul = sort!(unique(gc_label))
    cs = zeros(Int, length(ul))
    for i ∈ eachindex(ul)
        id = (gc_label .== ul[i])
        gc_label[id] .= i
        cs[i] = sum(gs[id]) # cluster size = sum of all group sizes that form cluster
    end
    return cs, gc_label, gc_half_nrm2
end

function min_pts!(label::Vector{Int}, gc::Vector{Int}, gs::Vector{Int}, cs::Vector{Int}, gc_label::Vector{Int}, gc_half_nrm2::Vector{Float64}, ind::Vector{Int}, group_label::Vector{Int}, minPts::Int)
    # Now eliminate tiny clusters by reassigning each of the constituting groups
    # to the nearest group belonging to a cluster with at least minPts points. 
    # This means, we are potentially dissolving tiny clusters, reassigning groups 
    # to different clusters.
    #
    # ! This function modifies label, gc, cs and gc_label !
    
    id = cs[cs .< minPts]   # cluster labels with small number of total points
    copy_gc_label = gc_label # added by Xinye (gc_label's before reassignment of tiny groups)
    d = zeros(length(gc_half_nrm2))
   
    for i ∈ id
        ii = i[copy_gc_label .== i] # find all tiny groups with that label
        for iii ∈ ii
            xi = view(gc_x,:,iii)        # group center (starting point) of one tiny group
            
            #d = gc_half_nrm2 - xi'*gc_x + gc_half_nrm2(iii); # half squared distance to all groups
            d .= @views gc_half_nrm2 .- xi'*gc_x                      # don't need the constant term
            
            o = sortperm(d)      # indices of group centers ordered by distance from gc_x(:,iii)
            for j ∈ o         # go through all of them in order and stop when a sufficiently large group has been found
                if cs[copy_gc_label[j]] >= minPts
                    gc_label[iii] = copy_gc_label[j]
                    break
                end
            end
        end
    end

    # rename labels to be 1,2,3,... and determine cluster sizes again
    # needs to be redone because the tiny groups have now disappeared
    ul = unique(gc_label)
    cs .= 0
    for i ∈ eachindex(ul)
        id = (gc_label .== ul[i])
        gc_label[id] .= i
        cs[i] = sum(gs[id])
    end
    
    # now relabel all labels, not just group centers
    label .= gc_label[label]
    
    # unsort data labels
    J = sortperm(ind)
    label .= label[J]
    group_label .= group_label[J]
    
    # unsort group centers
    gc .= ind[gc] 
    return nothing
end

function explain_fun(x::Matrix{Float64}, label::Vector{Int}, group_label::Vector{Int}, U::Matrix{Float64}, out, radius::Float64, minPts::Int)
    return "Explain function not implemented yet."
end

# precompile:
classix(randn(5,3))

end # module