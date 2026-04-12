using StatsBase

"""
Just an adapter for `StatsBase.entropy`, which works only for probability
distributions (i.e., values ranging between 0 and 1).
"""
function entropy(col)
    counts = values(countmap(col))
    probs = collect(counts) ./ sum(counts)
    return StatsBase.entropy(probs)
end

"""
Coerce the types of a generic dataframe, in such a way that they are suitable
to be trated by MLJ and Sole.jl.
"""
function coerce_dataframe(X::AbstractDataFrame)
    coerce_dict = Dict{Symbol, Type}()

    for col in names(X)
        sc = eltype(scitype(X[:, col]))

        # why do we want to do this?
        # because DecisionTree.jl does not support MultiClass;
        # we could one-hot encode the MultiClass columns, but many SHARE
        # questions are worded in a way for which an ordering is induced.
        if sc <: MultiClass
            coerce_dict[col] = OrderedFactor
        else
            coerce_dict[col] = sc
        end

		# coerce_dict[Symbol(col)] = eltype(scitype(X[:, col]))
    end

    return coerce(X, coerce_dict...)
end
