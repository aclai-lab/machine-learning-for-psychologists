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

	# necessary to read and write the work done in the previous notebook
    using Serialization

	# general utilities for data encodings, statistics and visualizations
    using DataFrames
    using StatsBase
    using PlutoUI
	import PlutoUI: combine
	using Plots

	# the machine learning engines!
    using MLJ
	using SoleModels

	# generics
	using Random
	INCLUDE_PATH = joinpath(@__DIR__, "..", "utils")
	include(joinpath(INCLUDE_PATH, "adapters.jl"));

	# just a flag to suppress some warnings
	scitype_check_level=0
end

# ╔═╡ 53c022c4-b0f3-42c0-94b0-7114bec855e7
# code to use to guarantee reproducibility when leveraging randomness
RNG_SEED = 1605

# ╔═╡ d7df4cf0-e938-471a-8284-e741588cf830
md"""
# Data Loading

In the previous notebook, we serialized the cleaned data.
Here, we load the same exact data.
"""

# ╔═╡ a2000000-3353-11f1-90b2-21952756a80b
begin
    DATASET_FOLDER = joinpath(@__DIR__, "..", "datasets")
    SERIALIZE_PATH = joinpath(DATASET_FOLDER, "share_clean.jls")
end

# ╔═╡ a3000000-3353-11f1-90b2-21952756a80b
data = deserialize(SERIALIZE_PATH)

# ╔═╡ cfc1ddfb-3171-41df-a319-7e55b6ac79ad
md"""
# Data adaptation
TODO: write about the fact that we are going to leverage MLJ and its scientific types system.
"""

# ╔═╡ f0753df6-d698-4649-b5fd-ac7dcb385ce8
md"""
!!! warning "Scientific Types"
	TODO: write about scientific types, that are fundamental to bridge statistics into machine learning.
"""

# ╔═╡ ea285fa0-cd96-41e2-a8bb-356fd2606eb7
y = coerce(string.(data[:, :euro_d]), Multiclass)

# ╔═╡ 1295b232-f44e-415e-b69b-b6f20786da58
X_raw = select(data, Not(:euro_d))

# ╔═╡ 9581203e-8089-4d8a-b25a-d3806a499cda
X_coerced = coerce_dataframe(X_raw)

# ╔═╡ a5000000-3353-11f1-90b2-21952756a80b
schema(X_coerced)

# ╔═╡ d4e2b317-92b9-4f59-b63d-91d14d3af828
md"""
TODO: describe what is happening here.
"""

# ╔═╡ a6000000-3353-11f1-90b2-21952756a80b
begin
    imputer = FillImputer()
    imp_mach = machine(imputer, X_coerced)
    fit!(imp_mach)
    X = MLJ.transform(imp_mach, X_coerced)
    schema(X)
end

# ╔═╡ d2d1e843-88ee-442b-a02f-1539d0ac514b
unique(X_coerced[:, "residence_rural_urban"])

# ╔═╡ 2107b65c-8af3-4914-a0c4-cbd2412bbab4
unique(X[:, "residence_rural_urban"])

# ╔═╡ a7000000-3353-11f1-90b2-21952756a80b
X_ninstances, X_nattributes = size(X)

# ╔═╡ b0000000-3353-11f1-90b2-21952756a80b
begin
    (X_train, X_test), (y_train, y_test) = partition(
        (X, y), 0.7;

		rng=RNG_SEED, 

		# sometimes, data is ordered via a criterion we do not want to assume
		shuffle=true,

		# we need this to keep Xs and ys glued pairwise
		multi=true
    )
end

# ╔═╡ 6abe7cf4-231c-4f75-839f-6b80891d3088
md"""
!!! warning "Unbalanced classes"
	TODO: explain why having unbalanced classes is dangerous.

	TODO: mention the fact that we are going to fix this later, using cross validation with stratified sampling.

	TODO: warn the student about how to read the bar plot below...
"""

# ╔═╡ c70c1204-60a1-4ecf-b0a1-8a60938686ff
y_train_yes, y_train_no = values(countmap(y_train));

# ╔═╡ d9bac238-70b3-43a0-95c5-99fb5ae96b92
y_test_yes, y_test_no = values(countmap(y_test));

# ╔═╡ 57d8be18-2370-4fbc-bc69-eb54898e9dff
bar(["train", "test"], 
	[[y_train_no, y_train_yes], [y_test_no, y_test_yes]],
	title="Train set class distribution",
	label=["no" "yes"])

# ╔═╡ 6b94d7db-7f2f-4471-8999-b138b6b0c448
md"""
# Training (first approach)

We proceed to leverage the training data to *induce a decision tree*.

The driver engine is MLJ, but the learning logic comes from a famous package of the Julia community (DecisionTreeClassifier).
"""

# ╔═╡ b1000000-3353-11f1-90b2-21952756a80b
begin
    try
        DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    catch
        println("DecisionTreeClassifier already imported.")
    end
end

# ╔═╡ c1ea8a00-0772-4717-af4b-cf848a825a10
possible_depths = collect(1:10)

# ╔═╡ 6d24d012-d0ba-4cfa-80a3-2126833c7f80
begin
	models = []
	for d in possible_depths
		model = MLJDecisionTreeInterface.DecisionTreeClassifier(
	    	max_depth         = d,
	    	min_samples_leaf  = 1,
	    	min_samples_split = 2
		)
		push!(models,model)
	end
end

# ╔═╡ b3000000-3353-11f1-90b2-21952756a80b
@bind max_depth_value Slider(possible_depths, default=5, show_value=true)

# ╔═╡ 6fa2ba7c-720a-46be-a2ac-4a28115b4d84
##TODO: spieghiamo varie robe 

# ╔═╡ b4000000-3353-11f1-90b2-21952756a80b
model = models[max_depth_value]

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

# ╔═╡ 31eb4188-2a5a-42f3-8533-c56c62f0bbaa
## VOGLIAMO ALLORA FARE PARTISION BILANCIATI 

# ╔═╡ b7000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Decision Tree (test set):** $(round(accuracy(cm_dt), digits=4))"

# ╔═╡ b8000000-3353-11f1-90b2-21952756a80b
begin
    mach_cv = machine(model, X, y)
    acc_cv = evaluate!(
        mach_cv;
        resampling = StratifiedCV(nfolds=10, shuffle=true),
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
         resampling = StratifiedCV(nfolds=10, shuffle=true),
         range      = [max_depth_range, min_samples_leaf_range, min_samples_split_range],
         measure    = accuracy,
         tuning     = RandomSearch()
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

# ╔═╡ 5117a6c9-b090-4d85-9cbc-9500beb09f7e
#a = SoleModels.solemodel(fitted_params(mach_dt).tree)

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

# ╔═╡ b6bc4920-7538-4df9-8295-4730db58be77
begin
	possible_tree_depths  = collect(1:10)
	possible_tree_numbers = collect(2:20)
end

# ╔═╡ b03265a4-0776-41c9-846a-5e74b459814d
function pollo(directions::Vector)
	
	return combine() do Child
		
		inputs = [
			md""" $(name): $(
				Child(name, Slider(1:100 , show_value = true))
			)"""
			
			for name in directions
		]
		
		md"""
		#### Wind speeds
		$(inputs)
		"""
	end
end

# ╔═╡ 1d0cc054-5c38-46d5-9033-4872d265c0d8
@bind trees_param pollo(["max_depth", "min_samples_leaf", "min_samples_split", "n_trees"]) 

# ╔═╡ c5000000-3353-11f1-90b2-21952756a80b
begin
    forest = MLJDecisionTreeInterface.RandomForestClassifier(
        max_depth         = trees_param.max_depth,
        min_samples_leaf  = trees_param.min_samples_leaf,
        min_samples_split = trees_param.min_samples_split,
        n_trees           = trees_param.n_trees
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

# ╔═╡ 8d28f8b0-1165-4ed2-bd3e-a09741363e83
md"""
# TODO
----
- SEPARARE BENE INPUTE NEL NOTEBOOK PRIMA 
- ORDINARE E COMMENTARE QUESTO CODICE  
- SOLEMODELS
- POSTHOC
"""

# ╔═╡ Cell order:
# ╠═a1000000-3353-11f1-90b2-21952756a80b
# ╠═53c022c4-b0f3-42c0-94b0-7114bec855e7
# ╟─d7df4cf0-e938-471a-8284-e741588cf830
# ╠═a2000000-3353-11f1-90b2-21952756a80b
# ╠═a3000000-3353-11f1-90b2-21952756a80b
# ╟─cfc1ddfb-3171-41df-a319-7e55b6ac79ad
# ╠═f0753df6-d698-4649-b5fd-ac7dcb385ce8
# ╠═ea285fa0-cd96-41e2-a8bb-356fd2606eb7
# ╠═1295b232-f44e-415e-b69b-b6f20786da58
# ╠═9581203e-8089-4d8a-b25a-d3806a499cda
# ╠═a5000000-3353-11f1-90b2-21952756a80b
# ╠═d4e2b317-92b9-4f59-b63d-91d14d3af828
# ╠═a6000000-3353-11f1-90b2-21952756a80b
# ╠═d2d1e843-88ee-442b-a02f-1539d0ac514b
# ╠═2107b65c-8af3-4914-a0c4-cbd2412bbab4
# ╠═a7000000-3353-11f1-90b2-21952756a80b
# ╠═b0000000-3353-11f1-90b2-21952756a80b
# ╠═6abe7cf4-231c-4f75-839f-6b80891d3088
# ╠═c70c1204-60a1-4ecf-b0a1-8a60938686ff
# ╠═d9bac238-70b3-43a0-95c5-99fb5ae96b92
# ╠═57d8be18-2370-4fbc-bc69-eb54898e9dff
# ╟─6b94d7db-7f2f-4471-8999-b138b6b0c448
# ╠═b1000000-3353-11f1-90b2-21952756a80b
# ╠═c1ea8a00-0772-4717-af4b-cf848a825a10
# ╠═6d24d012-d0ba-4cfa-80a3-2126833c7f80
# ╠═b3000000-3353-11f1-90b2-21952756a80b
# ╠═6fa2ba7c-720a-46be-a2ac-4a28115b4d84
# ╠═b4000000-3353-11f1-90b2-21952756a80b
# ╠═b5000000-3353-11f1-90b2-21952756a80b
# ╠═b6000000-3353-11f1-90b2-21952756a80b
# ╠═31eb4188-2a5a-42f3-8533-c56c62f0bbaa
# ╟─b7000000-3353-11f1-90b2-21952756a80b
# ╠═b8000000-3353-11f1-90b2-21952756a80b
# ╟─b9000000-3353-11f1-90b2-21952756a80b
# ╠═c0000000-3353-11f1-90b2-21952756a80b
# ╠═c1000000-3353-11f1-90b2-21952756a80b
# ╠═c2000000-3353-11f1-90b2-21952756a80b
# ╠═5117a6c9-b090-4d85-9cbc-9500beb09f7e
# ╟─c3000000-3353-11f1-90b2-21952756a80b
# ╠═c4000000-3353-11f1-90b2-21952756a80b
# ╠═b6bc4920-7538-4df9-8295-4730db58be77
# ╠═b03265a4-0776-41c9-846a-5e74b459814d
# ╠═1d0cc054-5c38-46d5-9033-4872d265c0d8
# ╠═c5000000-3353-11f1-90b2-21952756a80b
# ╠═c6000000-3353-11f1-90b2-21952756a80b
# ╟─c7000000-3353-11f1-90b2-21952756a80b
# ╟─c8000000-3353-11f1-90b2-21952756a80b
# ╠═8d28f8b0-1165-4ed2-bd3e-a09741363e83
