using DataFrames
using CategoricalArrays

"""
TODO: document
"""
function filter_along_dimension(
    df::Any,
    max_occurrences::Int;
    dims::Symbol=:cols,
    property=ismissing,
    colnames=[]
)
    if colnames == []
        colnames = names(df)
    end

    if dims == :cols
        miss_counts = [count(property, df[!, c]) for c in colnames]
        to_drop = findall(>=(max_occurrences), miss_counts)
        return select(df, Not(to_drop))
    elseif dims == :rows
        miss_counts = [
            count(property, (row[c] for c in colnames))
            for row in eachrow(df)
        ]
        to_drop = findall(>=(max_occurrences), miss_counts)
        return df[Not(to_drop), :]
    else
        error("dims must be :cols or :rows")
    end
end

function cast_columns(df; cat_threshold=10)
    for col in names(df)
        x = df[!, col]

        if eltype(x) <: Union{AbstractString,Bool,Missing}
            df[!, col] = categorical(x)
        elseif eltype(x) <: Real
            # if there are a few unique values, then this is categorical
            if length(unique(x)) <= cat_threshold
                df[!, col] = categorical(x)
            else
                df[!, col] = Float64.(x)
            end
        else
            # default case is categorical
            df[!, col] = categorical(string.(x))
        end
    end

    return df
end

function filter_variance()

end

function filter_threshold()

end

# """
# TODO: document
# """
# function make_categorical!(df; threshold=10)
#     for col in names(df)
#         v = df[!, col]
#
#         # Skip columns that are entirely missing
#         all(ismissing, v) && continue
#
#         # Work on non-missing values
#         vals = collect(skipmissing(v))
#         nunique = length(unique(vals))
#
#         if eltype(v) <: AbstractString || eltype(v) <: Bool
#             df[!, col] = categorical(v)
#
#         elseif eltype(v) <: Number && nunique <= threshold
#             df[!, col] = categorical(v)
#         end
#     end
#
#     return df
# end

