using DataFrames

function filter_along_dimension(
    df::Any,
    max_occurrences::Int;
    dims::Symbol = :cols,
    property = ismissing,
    colnames = []
)
    if colnames == []
        colnames = names(df)
    end

    if dims == :cols
        miss_counts = [count(property, df[!, c]) for c in colnames]
        to_drop = findall(>=(max_occurrences), miss_counts)
        return select(df, Not(to_drop))
    elseif dims == :rows
        miss_counts = [count(property, row) for row in eachrow(df)]
        to_drop = findall(>=(max_occurrences), miss_counts)
        return df[Not(to_drop), :]
    else
        error("dims must be :cols or :rows")
    end
end

