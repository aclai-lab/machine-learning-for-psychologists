### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a1000000-3353-11f1-90b2-21952756a80b
begin
    import Pkg
    Pkg.activate(Base.current_project(@__DIR__))

    using Serialization
    using DataFrames
    using MLJ
    using Plots
    using Random
    using StatsBase
    using PlutoUI
end

# ╔═╡ a2000000-3353-11f1-90b2-21952756a80b
begin
    DATASET_FOLDER = joinpath(@__DIR__, "..", "datasets")
    SERIALIZE_PATH = joinpath(DATASET_FOLDER, "share_clean.jls")
end

# ╔═╡ a3000000-3353-11f1-90b2-21952756a80b
data = deserialize(SERIALIZE_PATH)

# ╔═╡ a5000000-3353-11f1-90b2-21952756a80b
begin
    y = coerce(data[:, :euro_d], Multiclass)
    X_raw = select(data, Not(:euro_d))

    # Coerci automaticamente in base al tipo attuale
    coerce_dict = Dict{Symbol, Type}()
    for col in names(X_raw)
        st = scitype(X_raw[:, col])
        if st <: AbstractVector{<:Union{Missing, Multiclass}}
            coerce_dict[Symbol(col)] = Union{Missing, OrderedFactor}
        elseif st <: AbstractVector{<:Multiclass}
            coerce_dict[Symbol(col)] = OrderedFactor
        end
    end

    X_coerced = coerce(X_raw, coerce_dict...)
    schema(X_coerced)
end

# ╔═╡ a6000000-3353-11f1-90b2-21952756a80b
# SO CHE NOI GIA ARRIVIAMO DA PRE PROCESSING MA VOI LO FACEVATE
begin
    imputer = FillImputer()
    imp_mach = machine(imputer, X_coerced)
    fit!(imp_mach)
    X = MLJ.transform(imp_mach, X_coerced)
    schema(X)
end

# ╔═╡ a7000000-3353-11f1-90b2-21952756a80b
begin
    X_ninstances, X_nattributes = size(X)
end

# ╔═╡ a8000000-3353-11f1-90b2-21952756a80b
md"""
### Info su X
- Istanze: **$(X_ninstances)**
- Attributi: **$(X_nattributes)**
"""

# ╔═╡ a9000000-3353-11f1-90b2-21952756a80b
md"""
### Info su y
Classi: **$(unique(y))**
"""

# ╔═╡ b0000000-3353-11f1-90b2-21952756a80b
begin
    (X_train, X_test), (y_train, y_test) = partition(
        (X, y), 0.7;
        rng=121, shuffle=true, multi=true
    )
end

# ╔═╡ b1000000-3353-11f1-90b2-21952756a80b
begin
    try
        DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    catch
        println("DecisionTreeClassifier già importato.")
    end
end

# ╔═╡ b2000000-3353-11f1-90b2-21952756a80b
md"## Model Training"

# ╔═╡ b3000000-3353-11f1-90b2-21952756a80b
@bind max_depth_value Slider(2:10, default=5, show_value=true)

# ╔═╡ b4000000-3353-11f1-90b2-21952756a80b
model = MLJDecisionTreeInterface.DecisionTreeClassifier(
    max_depth         = max_depth_value,
    min_samples_leaf  = 1,
    min_samples_split = 2
)

# ╔═╡ b5000000-3353-11f1-90b2-21952756a80b
begin
    mach_dt = machine(model, X_train, y_train)
    fit!(mach_dt, verbosity=0)
    y_prob_dt = MLJ.predict(mach_dt, X_test)
    y_pred_dt = mode.(y_prob_dt)
    cm_dt     = confusion_matrix(y_pred_dt, y_test)
end

# ╔═╡ b6000000-3353-11f1-90b2-21952756a80b
cm_dt

# ╔═╡ b7000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Decision Tree (test set):** $(round(accuracy(cm_dt), digits=4))"

# ╔═╡ b8000000-3353-11f1-90b2-21952756a80b
begin
    mach_cv = machine(model, X, y)
    acc_cv = evaluate!(
        mach_cv;
        resampling = StratifiedCV(nfolds=5, shuffle=true),
        measures   = [accuracy],
        verbosity  = 0
    )
    acc_cv
end

# ╔═╡ b9000000-3353-11f1-90b2-21952756a80b
md"## Hyperparameter Tuning"

# ╔═╡ c0000000-3353-11f1-90b2-21952756a80b
begin
    max_depth_range         = range(model, :max_depth,          lower=2, upper=10)
    min_samples_leaf_range  = range(model, :min_samples_leaf,   lower=1, upper=5)
    min_samples_split_range = range(model, :min_samples_split,  lower=2, upper=10)

    tuned_tree = TunedModel(
        model      = MLJDecisionTreeInterface.DecisionTreeClassifier(),
        resampling = StratifiedCV(nfolds=5, shuffle=true),
        range      = [max_depth_range, min_samples_leaf_range, min_samples_split_range],
        measure    = accuracy,
        tuning     = Grid()
    )

    mach_tuned = machine(tuned_tree, X, y)
    fit!(mach_tuned, verbosity=0)

    y_prob_tuned = MLJ.predict(mach_tuned, X_test)
    y_pred_tuned = mode.(y_prob_tuned)
    cm_tuned     = confusion_matrix(y_pred_tuned, y_test)
end

# ╔═╡ c1000000-3353-11f1-90b2-21952756a80b
report(mach_tuned).best_model

# ╔═╡ c2000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Tuned DT (test set):** $(round(accuracy(cm_tuned), digits=4))"

# ╔═╡ c3000000-3353-11f1-90b2-21952756a80b
md"## Random Forest"

# ╔═╡ c4000000-3353-11f1-90b2-21952756a80b
begin
    try
        RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
    catch
        println("RandomForestClassifier già importato.")
    end
end

# ╔═╡ c5000000-3353-11f1-90b2-21952756a80b
begin
    forest = MLJDecisionTreeInterface.RandomForestClassifier(
        max_depth         = 3,
        min_samples_leaf  = 1,
        min_samples_split = 2,
        n_trees           = 10
    )

    mach_rf = machine(forest, X_train, y_train)
    fit!(mach_rf, verbosity=0)

    y_prob_rf = MLJ.predict(mach_rf, X_test)
    y_pred_rf = mode.(y_prob_rf)
    cm_rf     = confusion_matrix(y_pred_rf, y_test)
end

# ╔═╡ c6000000-3353-11f1-90b2-21952756a80b
cm_rf

# ╔═╡ c7000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Random Forest (test set):** $(round(accuracy(cm_rf), digits=4))"

# ╔═╡ c8000000-3353-11f1-90b2-21952756a80b
md"""
## Riepilogo accuratezze

| Modello | Accuracy |
|---------|----------|
| Decision Tree (depth=$(max_depth_value)) | $(round(accuracy(cm_dt), digits=4)) |
| Decision Tree Tuned | $(round(accuracy(cm_tuned), digits=4)) |
| Random Forest | $(round(accuracy(cm_rf), digits=4)) |
"""

# ╔═╡ Cell order:
# ╠═a1000000-3353-11f1-90b2-21952756a80b
# ╠═a2000000-3353-11f1-90b2-21952756a80b
# ╠═a3000000-3353-11f1-90b2-21952756a80b
# ╠═a5000000-3353-11f1-90b2-21952756a80b
# ╠═a6000000-3353-11f1-90b2-21952756a80b
# ╠═a7000000-3353-11f1-90b2-21952756a80b
# ╟─a8000000-3353-11f1-90b2-21952756a80b
# ╟─a9000000-3353-11f1-90b2-21952756a80b
# ╠═b0000000-3353-11f1-90b2-21952756a80b
# ╠═b1000000-3353-11f1-90b2-21952756a80b
# ╟─b2000000-3353-11f1-90b2-21952756a80b
# ╠═b3000000-3353-11f1-90b2-21952756a80b
# ╠═b4000000-3353-11f1-90b2-21952756a80b
# ╠═b5000000-3353-11f1-90b2-21952756a80b
# ╠═b6000000-3353-11f1-90b2-21952756a80b
# ╟─b7000000-3353-11f1-90b2-21952756a80b
# ╠═b8000000-3353-11f1-90b2-21952756a80b
# ╟─b9000000-3353-11f1-90b2-21952756a80b
# ╠═c0000000-3353-11f1-90b2-21952756a80b
# ╠═c1000000-3353-11f1-90b2-21952756a80b
# ╟─c2000000-3353-11f1-90b2-21952756a80b
# ╟─c3000000-3353-11f1-90b2-21952756a80b
# ╠═c4000000-3353-11f1-90b2-21952756a80b
# ╠═c5000000-3353-11f1-90b2-21952756a80b
# ╠═c6000000-3353-11f1-90b2-21952756a80b
# ╟─c7000000-3353-11f1-90b2-21952756a80b
# ╟─c8000000-3353-11f1-90b2-21952756a80b
