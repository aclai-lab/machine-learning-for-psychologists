using DataFrames
using CategoricalArrays
using StatsBase
using Statistics


_resolve_cols(df::AbstractDataFrame, colnames::AbstractVector) =
    isempty(colnames) ? names(df) : collect(String, colnames)


function _col_entropy(col)
    clean = collect(skipmissing(col))
    isempty(clean) && return 0.0

    counts = countmap(clean)
    n = sum(values(counts))
    return -sum(c -> (p = c / n; p * log(p)), values(counts))
end

filter_df(df::AbstractDataFrame, mode::Symbol; kwargs...) =
    filter_df(df, Val(mode); kwargs...)

# Friendly error for unknown modes
function filter_df(df::AbstractDataFrame, ::Val{M}; kwargs...) where {M}
    throw(ArgumentError("""
        Unknown filter mode: $(repr(M)).
        Valid modes: :missing_cols, :missing_rows, :property_cols,
        :property_rows, :frequency, :information, :zscore, :cast
    """))
end

################################################################################

function filter_df(df::AbstractDataFrame, ::Val{:missing_cols};
    max_missing::Int,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = filter(c -> c ∉ ignore_cols, _resolve_cols(df, colnames))
    bad = filter(c -> count(ismissing, df[!, c]) > max_missing, cols)
    return isempty(bad) ? copy(df) : select(df, Not(bad))
end

function filter_df(df::AbstractDataFrame, ::Val{:missing_rows};
    max_missing::Int,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = filter(c -> c ∉ ignore_cols, _resolve_cols(df, colnames))
    keep = [count(ismissing, (row[c] for c in cols)) <= max_missing
            for row in eachrow(df)]
    return df[keep, :]
end

function filter_df(df::AbstractDataFrame, ::Val{:property_cols};
    max_occurrences::Int,
    property=ismissing,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = filter(c -> c ∉ ignore_cols, _resolve_cols(df, colnames))
    bad = filter(c -> count(property, df[!, c]) >= max_occurrences, cols)
    return isempty(bad) ? copy(df) : select(df, Not(bad))
end

function filter_df(df::AbstractDataFrame, ::Val{:property_rows};
    max_occurrences::Int,
    property=ismissing,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = filter(c -> c ∉ ignore_cols, _resolve_cols(df, colnames))
    keep = [count(property, (row[c] for c in cols)) < max_occurrences
            for row in eachrow(df)]
    return df[keep, :]
end

function filter_df(df::AbstractDataFrame, ::Val{:frequency};
    frequency_threshold::Real=0.6,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = filter(c -> c ∉ ignore_cols, _resolve_cols(df, colnames))
    bad = filter(cols) do c
        vals = collect(skipmissing(df[!, c]))
        isempty(vals) && return false
        maximum(values(countmap(vals))) / length(vals) >= frequency_threshold
    end
    return isempty(bad) ? copy(df) : select(df, Not(bad))
end

function filter_df(df::AbstractDataFrame, ::Val{:information};
    information_dictionary=Dict{String,Float64},
    information_threshold::Real=0.5,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = filter(c -> c ∉ ignore_cols, _resolve_cols(df, colnames))
    bad = filter(c -> information_dictionary[c] < information_threshold, cols)
    return isempty(bad) ? copy(df) : select(df, Not(bad))
end

function filter_df(df::AbstractDataFrame, ::Val{:zscore};
    z_threshold::Real=1.65,
    colnames::AbstractVector=String[],
    ignore_cols::AbstractVector=String[])
    cols = if isempty(colnames)
        filter(c -> c ∉ ignore_cols && eltype(df[!, c]) <: Union{Real,Missing}, names(df))
    else
        filter(c -> c ∉ ignore_cols, collect(String, colnames))
    end

    keep = trues(nrow(df))
    for c in cols
        col = df[!, c]
        μ = mean(skipmissing(col))
        σ = std(skipmissing(col))
        σ == 0 && continue
        keep .&= [ismissing(v) || abs((v - μ) / σ) <= z_threshold for v in col]
    end
    return df[keep, :]
end

function filter_df(df::AbstractDataFrame,
    ::Val{:cast};
    cast_threshold::Int=30,
    ignore_cols::AbstractVector=String[]
)
    out = deepcopy(df)

    for col in names(out)
        col ∈ ignore_cols && continue

        x = out[!, col]

        nof_unique_values = length(unique(x))

        if nof_unique_values <= cast_threshold
            out[!, col] = categorical(x)
        else
            # keep out[!, col] as a Float64?
        end

    #
    #     T = eltype(x)
    #
    #     if T <: Union{AbstractString,Bool,Missing}
    #         out[!, col] = categorical(x)
    #     elseif T <: Union{Real,Missing}
    #         if length(unique(skipmissing(x))) <= cast_threshold
    #             out[!, col] = categorical(x)
    #         else
    #             out[!, col] = passmissing(Float64).(x)
    #         end
    #     else
    #         out[!, col] = categorical(string.(x))
    #     end
    # end
    return out
end
