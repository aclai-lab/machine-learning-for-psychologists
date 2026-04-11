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

