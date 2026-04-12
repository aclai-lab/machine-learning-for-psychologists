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
	using MLJBase
	using MLJTransforms
	using SoleModels

	# generics
	using Random
	INCLUDE_PATH = joinpath(@__DIR__, "..", "utils")
	include(joinpath(INCLUDE_PATH, "adapters.jl"));
	include(joinpath(INCLUDE_PATH, "measures.jl"));

	# just a flag to suppress some warnings
	scitype_check_level=0;
end

# ╔═╡ 34bafc6f-ac2a-4cdb-b9c2-f766111251cb
md"""
# SHARE Training Pipeline

In this notebook, we are going to leverage **MLJ** and **Sole** to train decision trees and forests for learning and manipulating the theory underlying the SHARE dataset.

MLJ is probably the most famous package in Julia for supporting machine learning workflows with structured (i.e., tabular) data.

Sole is a framework specialized for the treatment of *symbolic* models. Very briefly, it enables the learning of original models, a deep inspection and optimization of the latter, and even dealing with unstructured (i.e., non tabular) data.

It is very common to see "double-connected channels" between MLJ and other packages of the Julia community! As we shall later, Sole is no exception, and the two frameworks can be used in synergy.
"""

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
    SERIALIZE_PATH = joinpath(DATASET_FOLDER, "share_clean_test_checkpoint.jls")
end

# ╔═╡ a3000000-3353-11f1-90b2-21952756a80b
data = begin
	try
		deserialize(SERIALIZE_PATH)
	catch 
		println("Something went wrong, maybe you need to change the file path?")
	end
end


# ╔═╡ cfc1ddfb-3171-41df-a319-7e55b6ac79ad
md"""
# Data adaptation
In the following notebook, we are going to implement the learning process of a machine learning model (in particular, decision trees and forests).

The data exploration and cleaning process lives on its own, but we need to adapt the DataFrame we just deserialized in order to work with MLJ.

By design, MLJ needs us to convert the attribute types from standard Julia types to the most performant **ScientificTypes**.
"""

# ╔═╡ f0753df6-d698-4649-b5fd-ac7dcb385ce8
md"""
!!! warning "Scientific Types"
	**ScientificTypes.jl** is a very popular and light-weight Julia package, defining a collection of types for implementing conventions about the scientific interpretation of data.

	It makes a clear distinction between the *machine type* of the Julia programming language and the *scientific type*, which reflects how one object should be *interpreted*.

	For our use-case, the *Multiclass* and *Continuous* types are enough. For a list of all the available types, see [the Scientific Types documentation](https://juliaai.github.io/ScientificTypes.jl/dev/reference/#Reference).
"""

# ╔═╡ ea285fa0-cd96-41e2-a8bb-356fd2606eb7
# ╠═╡ disabled = true
#=╠═╡

  ╠═╡ =#

# ╔═╡ d3fdbf72-79ee-4712-8469-4d29768b4559
y = coerce(string.(data[:, :euro_d]), Multiclass)
# y = categorical(data[:, "euro_d"], ordered=true, levels=["no", "yes"])

# ╔═╡ 1295b232-f44e-415e-b69b-b6f20786da58
X_raw = DataFrames.select(data, Not(:euro_d))

# ╔═╡ 9581203e-8089-4d8a-b25a-d3806a499cda
X_coerced = coerce_dataframe(X_raw)

# ╔═╡ a5000000-3353-11f1-90b2-21952756a80b
schema(X_coerced)

# ╔═╡ 9f0ae520-5f94-4e34-bb03-f8c68a61157a
size(X_coerced)

# ╔═╡ d4e2b317-92b9-4f59-b63d-91d14d3af828
md"""
MLJ automatically inferred the correct scientific types for each attribute, and applied the conversion using the *coerce* function. 

As we can see above, however, the Missing type is kept separated from Multiclass, when specifying the type of a categorical value. 

Since the Multiclass scientific type explicits that there is no ordering between the values of an attribute, we can safely convert missings to a numerical value.
"""

# ╔═╡ a6000000-3353-11f1-90b2-21952756a80b
begin
    imputer = FillImputer()
    mach = machine(imputer, X_coerced)
    fit!(mach)
    X_coerced_raw = MLJ.transform(mach, X_coerced)
    schema(X_coerced_raw);
end

# ╔═╡ 66ec88be-2014-40a1-a221-e8fbed52c3b7
md"""
!!! warning "The danger of Multiclass"
	TODO explain about one hot encoding
"""

# ╔═╡ 67a333d7-4aa7-45bd-ad05-957fc87102f2
OneHotEncoder = @load OneHotEncoder pkg=MLJTransforms

# ╔═╡ 51278f4e-d554-48f6-a63f-14f3e78fee17
ohe_mach = machine(OneHotEncoder(), X_coerced_raw)

# ╔═╡ cf3b8afb-f32d-42f1-aa7e-7dfc0ec0ef17
ohe_mach_fitted = fit!(ohe_mach);

# ╔═╡ 4ccc6d8a-a6c3-42c5-bdcd-e489bcdb6798
X = MLJBase.transform(ohe_mach_fitted, X_coerced_raw);

# ╔═╡ 565c7e32-c824-41e8-a8ea-4d1e4a638c86
size(X)

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
	If some labels/classes appear much more frequently than others, the model may become biased toward predicting the majority class. 

	In that case, a naive classifier can achieve deceptively high accuracy simply by always predicting the most common label.

	Later, we are going to leverage a smarter technique to partition data, and we will we will combine accuracy with metrics that are not thrown off by class imbalance.
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

Actually, we define 10 different trees with different settings of the hyperparameters; then, we select one specific tree with a slider and proceed to train it.

Note that the settings we propose here are trivial: we only change the max_depth of each tree, from one to ten. Later, we are going to make the training pipeline more robust, exploring different parameterizations automatically
"""

# ╔═╡ 64d0496d-51f0-48f1-9068-f34328d7a857
md"""
!!! success "Decision trees"
	A decision tree makes predictions by applying a sequence of if–else rules on the attribute values.

	- Each *internal node* contains a condition (e.g., age > 65);
	- each *branch* corresponds to the outcome of that condition (i.e., true or false);
	- each *leaf* node contains the final prediction (e.g., the string "yes" or "no").

	Inducing a decision tree means to compute the most convenient attribute-value pairs onto which to perform the various split and their ordering. 

	All the induction is governed by measuring how much "information is gained" when choosing a certain split; that notion can be intimately connected to the information entropy measure.
"""

# ╔═╡ 76edf51c-7ca6-49d1-85e8-bab1b77c04c2
LocalResource("../images/decision_tree.png")

# ╔═╡ b1000000-3353-11f1-90b2-21952756a80b
begin
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
end

# ╔═╡ c1ea8a00-0772-4717-af4b-cf848a825a10
possible_max_depths = collect(1:10)

# ╔═╡ ebe914ce-c030-4bb6-bee7-f72713da1954
models = DecisionTreeClassifier[]

# ╔═╡ df689893-a895-4b3e-85a2-5364274bf575
for d in possible_max_depths
	model = MLJDecisionTreeInterface.DecisionTreeClassifier(
    	max_depth = d,
    	min_samples_leaf = 1,
    	min_samples_split = 2,
		min_purity_increase = 0.0,
		n_subfeatures = 0.0,
		post_prune = false,
		merge_purity_threshold = 0.9
	)
	push!(models,model)
end

# ╔═╡ b3000000-3353-11f1-90b2-21952756a80b
@bind max_depth_value Slider(possible_max_depths, default=5, show_value=true)

# ╔═╡ b4000000-3353-11f1-90b2-21952756a80b
model = models[max_depth_value]

# ╔═╡ b5000000-3353-11f1-90b2-21952756a80b
begin
    mach_dt = machine(model, X_train, y_train)
    fit!(mach_dt)
    y_prob_dt = MLJ.predict(mach_dt, X_test)
    y_pred_dt = mode.(y_prob_dt)
    cm_dt = confusion_matrix(y_pred_dt, y_test)
end

# ╔═╡ 38110cbb-592d-474b-bb6a-671e26cb2298
md"""
!!! info "How to read the confusion matrix"
	TODO: write about precision and recall.
"""

# ╔═╡ b6000000-3353-11f1-90b2-21952756a80b
cm_dt

# ╔═╡ b7000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Decision Tree (test set):** $(round(accuracy(cm_dt), digits=4))"

# ╔═╡ edc4f8f8-12b9-45b2-bbbe-32736dc6fbd3
md"**Precision Decision Tree (test set):** $(round(precision(cm_dt), digits=4))"

# ╔═╡ 948beef6-c940-4852-9500-76fb521aa437
md"**Recall Decision Tree (test set):** $(round(recall(cm_dt), digits=4))"

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
md"""
# Hyperparameter Tuning

TODO: explain about tuning models and grid search
"""

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

# ╔═╡ 5b76d562-5472-416b-b63c-1fef7c81cd5f
md"**Precision Tuned DT (test set):** $(round(precision(cm_tuned), digits=4))"

# ╔═╡ 58ee4d37-9fa9-4f09-9d37-15d9f69d5ac2
md"**Recall Tuned DT (test set):** $(round(recall(cm_tuned), digits=4))"

# ╔═╡ 009f7612-9336-4afb-9907-4d40934e2845
precision(cm_tuned)

# ╔═╡ 98213623-eaec-4928-bfea-c0f0c4f04f90
md"""
!!! tip "☀️ Model inspection with Sole ☀️"
	Describe a typical path in the decision tree below.
"""

# ╔═╡ 5117a6c9-b090-4d85-9cbc-9500beb09f7e
SoleModels.solemodel(fitted_params(mach_dt).tree)

# ╔═╡ c3000000-3353-11f1-90b2-21952756a80b
md"## Random Forest"

# ╔═╡ c4000000-3353-11f1-90b2-21952756a80b
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree

# ╔═╡ 9410b8b8-10f0-460b-b46c-a7715cee1fe2
possible_tree_depths  = collect(1:10)

# ╔═╡ b6bc4920-7538-4df9-8295-4730db58be77
possible_tree_numbers = collect(2:20)

# ╔═╡ b03265a4-0776-41c9-846a-5e74b459814d
function set_hyperparameters(directions::Vector)
	return combine() do Child
		inputs = [
			if name == "max_depth" || name == "n_trees"
				md""" $(name): $(
					Child(name, Slider(1:100 , show_value = true, default=10))
				)"""
			else
				md""" $(name): $(
					Child(name, Slider(1:10, show_value = true, default=3))
				)"""
			end
			
			for name in directions
		]
		
		md"""
		#### Forest hyperparameters
		$(inputs)
		"""
	end
end

# ╔═╡ 1d0cc054-5c38-46d5-9033-4872d265c0d8
@bind trees_param set_hyperparameters(["max_depth", "min_samples_leaf", "min_samples_split", "n_trees"]) 

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
md"**Accuracy Random Forest:** $(round(accuracy(cm_rf), digits=4))"

# ╔═╡ ca74043d-78ae-47a1-a58a-a8d349313fda
md"**Precision Random Forest:** $(round(precision(cm_rf), digits=4))"

# ╔═╡ f147f0f9-5e64-4413-9142-c0ec6d081506
md"**Recall Random Forest:** $(round(recall(cm_rf), digits=4))"

# ╔═╡ bf81a25b-cbbe-4f61-8fe2-d558f13aeb5d
md"""
| Modello | Accuracy | Precision | Recall |
|---------|----------|-----------|--------|
| Decision Tree (depth=$(max_depth_value)) | $(round(accuracy(cm_dt), digits=4)) | $(round(precision(cm_dt), digits=4)) | $(round(recall(cm_dt), digits=4)) |
| Decision Tree Tuned | $(round(accuracy(cm_tuned), digits=4)) | $(round(precision(cm_tuned), digits=4)) | $(round(recall(cm_tuned), digits=4)) |
| Random Forest | $(round(accuracy(cm_rf), digits=4)) | $(round(precision(cm_rf), digits=4)) | $(round(recall(cm_rf), digits=4)) |
"""

# ╔═╡ 6fa4955d-c170-4e43-8cf6-0158cc08f60a
 begin
     df_max_depth_range = range(model, :max_depth,          lower=2, upper=10)
     df_min_samples_leaf_range = range(model, :min_samples_leaf,   lower=1, upper=5)
     df_min_samples_split_range = range(model, :min_samples_split,  lower=2, upper=10)
 
     tuned_forest = TunedModel(
         model      = MLJDecisionTreeInterface.DecisionForest(),
         resampling = StratifiedCV(nfolds=10, shuffle=true),
         range      = [df_max_depth_range, df_min_samples_leaf_range, df_min_samples_split_range],
         measure    = accuracy,
         tuning     = RandomSearch()
     )
 
     df_mach_tuned = machine(tuned_forest, X, y)
     fit!(df_mach_tuned, verbosity=0)
 
     df_y_prob_tuned = MLJ.predict(df_mach_tuned, X_test)
     df_y_pred_tuned = mode.(y_prob_tuned)
     df_cm_tuned     = confusion_matrix(y_pred_tuned, y_test)
 end

# ╔═╡ 8d28f8b0-1165-4ed2-bd3e-a09741363e83
md"""
# TODO
----
- SOLEMODELS
- POSTHOC
- wrap MLJ and DecisionTree code in utils/adapters.jl when it makes sense to do it
"""

# ╔═╡ Cell order:
# ╟─34bafc6f-ac2a-4cdb-b9c2-f766111251cb
# ╠═a1000000-3353-11f1-90b2-21952756a80b
# ╠═53c022c4-b0f3-42c0-94b0-7114bec855e7
# ╟─d7df4cf0-e938-471a-8284-e741588cf830
# ╠═a2000000-3353-11f1-90b2-21952756a80b
# ╠═a3000000-3353-11f1-90b2-21952756a80b
# ╟─cfc1ddfb-3171-41df-a319-7e55b6ac79ad
# ╟─f0753df6-d698-4649-b5fd-ac7dcb385ce8
# ╠═ea285fa0-cd96-41e2-a8bb-356fd2606eb7
# ╠═d3fdbf72-79ee-4712-8469-4d29768b4559
# ╠═1295b232-f44e-415e-b69b-b6f20786da58
# ╠═9581203e-8089-4d8a-b25a-d3806a499cda
# ╠═a5000000-3353-11f1-90b2-21952756a80b
# ╠═9f0ae520-5f94-4e34-bb03-f8c68a61157a
# ╟─d4e2b317-92b9-4f59-b63d-91d14d3af828
# ╠═a6000000-3353-11f1-90b2-21952756a80b
# ╠═66ec88be-2014-40a1-a221-e8fbed52c3b7
# ╠═67a333d7-4aa7-45bd-ad05-957fc87102f2
# ╠═51278f4e-d554-48f6-a63f-14f3e78fee17
# ╠═cf3b8afb-f32d-42f1-aa7e-7dfc0ec0ef17
# ╠═4ccc6d8a-a6c3-42c5-bdcd-e489bcdb6798
# ╠═565c7e32-c824-41e8-a8ea-4d1e4a638c86
# ╠═a7000000-3353-11f1-90b2-21952756a80b
# ╠═b0000000-3353-11f1-90b2-21952756a80b
# ╟─6abe7cf4-231c-4f75-839f-6b80891d3088
# ╠═c70c1204-60a1-4ecf-b0a1-8a60938686ff
# ╠═d9bac238-70b3-43a0-95c5-99fb5ae96b92
# ╠═57d8be18-2370-4fbc-bc69-eb54898e9dff
# ╟─6b94d7db-7f2f-4471-8999-b138b6b0c448
# ╟─64d0496d-51f0-48f1-9068-f34328d7a857
# ╠═76edf51c-7ca6-49d1-85e8-bab1b77c04c2
# ╠═b1000000-3353-11f1-90b2-21952756a80b
# ╠═c1ea8a00-0772-4717-af4b-cf848a825a10
# ╠═ebe914ce-c030-4bb6-bee7-f72713da1954
# ╠═df689893-a895-4b3e-85a2-5364274bf575
# ╠═b3000000-3353-11f1-90b2-21952756a80b
# ╠═b4000000-3353-11f1-90b2-21952756a80b
# ╠═b5000000-3353-11f1-90b2-21952756a80b
# ╠═38110cbb-592d-474b-bb6a-671e26cb2298
# ╠═b6000000-3353-11f1-90b2-21952756a80b
# ╟─b7000000-3353-11f1-90b2-21952756a80b
# ╟─edc4f8f8-12b9-45b2-bbbe-32736dc6fbd3
# ╟─948beef6-c940-4852-9500-76fb521aa437
# ╠═b8000000-3353-11f1-90b2-21952756a80b
# ╠═b9000000-3353-11f1-90b2-21952756a80b
# ╠═c0000000-3353-11f1-90b2-21952756a80b
# ╠═c1000000-3353-11f1-90b2-21952756a80b
# ╟─c2000000-3353-11f1-90b2-21952756a80b
# ╠═5b76d562-5472-416b-b63c-1fef7c81cd5f
# ╠═58ee4d37-9fa9-4f09-9d37-15d9f69d5ac2
# ╠═009f7612-9336-4afb-9907-4d40934e2845
# ╠═98213623-eaec-4928-bfea-c0f0c4f04f90
# ╠═5117a6c9-b090-4d85-9cbc-9500beb09f7e
# ╟─c3000000-3353-11f1-90b2-21952756a80b
# ╠═c4000000-3353-11f1-90b2-21952756a80b
# ╠═9410b8b8-10f0-460b-b46c-a7715cee1fe2
# ╠═b6bc4920-7538-4df9-8295-4730db58be77
# ╠═b03265a4-0776-41c9-846a-5e74b459814d
# ╠═1d0cc054-5c38-46d5-9033-4872d265c0d8
# ╠═c5000000-3353-11f1-90b2-21952756a80b
# ╠═c6000000-3353-11f1-90b2-21952756a80b
# ╠═c7000000-3353-11f1-90b2-21952756a80b
# ╠═ca74043d-78ae-47a1-a58a-a8d349313fda
# ╠═f147f0f9-5e64-4413-9142-c0ec6d081506
# ╠═bf81a25b-cbbe-4f61-8fe2-d558f13aeb5d
# ╠═6fa4955d-c170-4e43-8cf6-0158cc08f60a
# ╠═8d28f8b0-1165-4ed2-bd3e-a09741363e83
