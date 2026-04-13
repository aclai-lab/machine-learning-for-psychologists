using StatisticalMeasures

function precision(cm::T) where {T<:StatisticalMeasures.ConfusionMatrices.ConfusionMatrix}
    m = cm.mat
    tp = m[2, 2]
    fp = m[1, 2]
    return tp / (tp + fp)
end

function recall(cm::T) where {T<:StatisticalMeasures.ConfusionMatrices.ConfusionMatrix}
    m = cm.mat
    tp = m[2, 2]
    fn = m[2, 1]
    return tp / (tp + fn)
end
