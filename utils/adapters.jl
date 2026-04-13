using StatsBase

"""
Perform binning if the given collection is not categorical.
"""
function maybe_bin(x; nbins=10)
    # do nothing if the collection is already categorical
    if !(eltype(x) <: Number)
        return x
    end

    ux = unique(x)
    if length(ux) ≤ nbins
        return x
    end

    edges = quantile(x, range(0, 1; length=nbins+1))
    return cut(x, edges; labels=1:nbins, extend=true)
end

"""
This is just an adapter for `StatsBase.entropy`, which works only for 
probability distributions (i.e., values in [0, 1] and summing to 1).
"""
function entropy(col)
    counts = values(countmap(col))
    probs = collect(counts) ./ sum(counts)
    return StatsBase.entropy(probs)
end

"""
See [`mutual_information`](@ref).
"""
function joint_entropy(x, y)
    joint_counts = values(countmap(zip(x, y)))
    probs = collect(joint_counts) ./ sum(joint_counts)
    return StatsBase.entropy(probs)
end

"""
Computes the mutual information I(X,Y) = H(Y) - H(Y | X) = H(X) + H(Y) - H(X|Y)
where H is the [`entropy`](@ref) function.
"""
function mutual_information(x, y; nbins=10)
    x_bin = maybe_bin(x; nbins=nbins)
    y_bin = maybe_bin(y; nbins=nbins)

    hx = entropy(x_bin)
    hy = entropy(y_bin)
    hxy = joint_entropy(x_bin, y_bin)

    return hx + hy - hxy
end

"""
Coerce the types of a generic dataframe, in such a way that they are suitable
to be trated by MLJ and Sole.jl.
"""
function coerce_dataframe(X::AbstractDataFrame)
    coerce_dict = Dict{Symbol, Type}()

    for col in names(X)
		coerce_dict[Symbol(col)] = eltype(scitype(X[:, col]))
    end

    return coerce(X, coerce_dict...)
end
