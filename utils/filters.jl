using DataFrames

function filter_along_dimension(
    df::DataFrame,
    max_missing::Int;
    dims::Symbol = :cols 
)
    if dims == :cols
        miss_counts = [count(ismissing, df[!, c]) for c in names(df)]
        to_drop = findall(>=(max_missing), miss_counts)
        return select(df, Not(to_drop))
    elseif dims == :rows
        miss_counts = [count(ismissing, row) for row in eachrow(df)]
        to_drop = findall(>=(max_missing), miss_counts)
        return df[Not(to_drop), :]
    else
        error("dims must be :cols or :rows")
    end
end

