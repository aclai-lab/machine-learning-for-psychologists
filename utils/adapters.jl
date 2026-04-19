using SoleLogics
using StatsBase

"""
Perform binning if the given collection is not categorical.
"""
function maybe_bin(x; nbins=10)
    # if data is categorical, do nothing
    if !(eltype(x) <: Number)
        return x
    end

    ux = unique(x)
    
    # if data is a float collection but there are just a few unique values,
    # treat it as a categorical
    if length(ux) ≤ nbins
        return x
    end

    qs = range(0, 1; length=nbins + 1)
    edges = unique(quantile(x, qs))

    # ensure the edges of the binning are valid
    if length(edges) ≤ 2
        return x
    end

    # assign bin labels 
    return cut(x, edges; labels=1:(length(edges)-1), extend=true)
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
    coerce_dict = Dict{Symbol,Type}()

    for col in names(X)
        coerce_dict[Symbol(col)] = eltype(scitype(X[:, col]))
    end

    return coerce(X, coerce_dict...)
end

"""
Replace all the occurrences of [`SoleLogics.CONJUNCTION`](@ref) and 
[`SoleLogics.DISJUNCTION`](@ref) in the [`SoleLogics.syntaxstring`](@ref) of 
`lm`, with "and" and "or".
"""
function pretty_print(lm::Any)
    left  = antecedent(lm)
    right = syntaxstring(consequent(lm))
    s = right
    clean = replace(s, r"\e\[[0-9;]*m" => "")
    clean = replace(clean, r"[^\w\s]" => "")
    clean = strip(clean)   # ← rimuove \n e spazi iniziali/finali

    result = match(r"\b(yes|no)\b", clean)
    result === nothing ? nothing : result.match

    _st = syntaxstring(left)
    return ("IF " * replace(_st, "∧"=>" and ", "∨" => " or ") * " THEN classify " * clean)
end



function pretty_print_decision_set(ds)
    HTML(join(
    ["<p>" * pretty_print(r) * "</p>" for r in rules(ds)],
    "\n"
))
end



