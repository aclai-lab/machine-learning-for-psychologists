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
	#using SoleData.Artifacts
	using SoleModels
	using SolePostHoc

	# generics
	using Random
	INCLUDE_PATH = joinpath(@__DIR__, "..", "utils")
	include(joinpath(INCLUDE_PATH, "adapters.jl"));
	include(joinpath(INCLUDE_PATH, "measures.jl"));

	# just a flag to suppress some warnings
	scitype_check_level=0;
end

# ╔═╡ 3fb5d980-1477-4a63-b691-03e2b8f9e5da
using CategoricalArrays

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

# ╔═╡ 9e1b6b33-65e7-4eb2-a26b-b622546a2d75
md"""
Artifacts are Julia "containers of data" that are not Julia packages.

SolePostHoc will leverage external programs for supporting advanced data compression functionalities.
"""

# ╔═╡ 8afd5b7b-d0d7-4dcb-8911-f0fe02bd07e4
#begin
#	fillartifacts()
#	abc_loader = ABCLoader()
#	mit_loader = MITESPRESSOLoader()
#end

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

# ╔═╡ d3fdbf72-79ee-4712-8469-4d29768b4559
y = coerce(string.(data[:, :euro_d]), Multiclass)
# y = categorical(data[:, "euro_d"], ordered=true, levels=["no", "yes"])

# ╔═╡ 43815971-fda4-4b32-abd6-77ee7df1e1d6
md"""
It is better to separate the rest of the data, since they require more treatment.
"""

# ╔═╡ 1295b232-f44e-415e-b69b-b6f20786da58
X_raw = DataFrames.select(data, Not(:euro_d))

# ╔═╡ 9f0ae520-5f94-4e34-bb03-f8c68a61157a
size(X_raw)

# ╔═╡ 7d268051-0758-4c20-ae25-253a2a4627e8
schema(X_raw)

# ╔═╡ d4e2b317-92b9-4f59-b63d-91d14d3af828
md"""
MLJ automatically inferred the correct scientific types for each attribute, and applied the conversion using the *coerce* function. 

As we can see above, however, the Missing type is kept separated from Multiclass, when specifying the type of a categorical value. 

Since the Multiclass scientific type explicits that there is no ordering between the values of an attribute, we can safely convert missings to a numerical value.
"""

# ╔═╡ 2ac0e4ca-7e77-47b8-b1b9-9e1ed0d1c426
md"""
!!! info "Exercise"
	In your opinion, why missings are kept separated from the Multiclass specifier?
"""

# ╔═╡ a6000000-3353-11f1-90b2-21952756a80b
begin
    imputer = FillImputer()
    mach = machine(imputer, X_raw)
    fit!(mach)
    X_coerced_raw = MLJ.transform(mach, X_raw)
    schema(X_coerced_raw);
end

# ╔═╡ 66ec88be-2014-40a1-a221-e8fbed52c3b7
md"""
!!! tips "From Multiclass to One-hot encoding"
	Multiclass labels are typically stored as categorical values, and the model must be able to interpret them correctly as distinct classes rather than arbitrary text or numbers.

	**It is by no means certain that a model can naturally handle Multiclass**.

	A safe, general solution is... **one-hot encoding**!

	It is a way of representing each category as a separate binary feature, where only one feature is "active" (i.e., set to 1) for a given observation and all others are 0. 

	With one-hot encoding...
	- we avoid introducing a false notion of ordering between the values of an attribute (as in the case of leveraging Multiclass types);
	- it is not mandatory for the trained model to be designed for handling categorical attributes.
"""

# ╔═╡ d0b073f2-2bba-433d-afc4-c5cc085ada62
LocalResource("../images/onehot_encoding.png")

# ╔═╡ cdcbdbae-6867-453f-bbae-c7490a2c3df2
md"""
In a few cells, we are going to play with a particular kind of machine learning model called *decision tree*; let us see if the implementation we are going to leverage supports the Multiclass scientific type by design.
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

# ╔═╡ d8b09a29-2f37-4ff5-af0f-b7c03258c8af
# TODO: show that the DecisionTree can be used with X_train, but not with X_raw
MLJ.models(matching(X_train, y_train))

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
 #begin
 #    max_depth_range         = range(model, :max_depth,          lower=2, upper=10)
 #    min_samples_leaf_range  = range(model, :min_samples_leaf,   lower=1, upper=5)
 #    min_samples_split_range = range(model, :min_samples_split,  lower=2, upper=10)
 #
 #    tuned_tree = TunedModel(
 #        model      = MLJDecisionTreeInterface.DecisionTreeClassifier(),
 #        resampling = StratifiedCV(nfolds=10, shuffle=true),
 #        range      = [max_depth_range, min_samples_leaf_range, #min_samples_split_range],
 #        measure    = accuracy,
 #        tuning     = RandomSearch()
 #    )
 #
 #    mach_tuned = machine(tuned_tree, X, y)
 #    fit!(mach_tuned, verbosity=0)
 #
 #    y_prob_tuned = MLJ.predict(mach_tuned, X_test)
 #    y_pred_tuned = mode.(y_prob_tuned)
 #    cm_tuned     = confusion_matrix(y_pred_tuned, y_test)
 #end

# ╔═╡ c1000000-3353-11f1-90b2-21952756a80b
#report(mach_tuned).best_model

# ╔═╡ c2000000-3353-11f1-90b2-21952756a80b
#md"**Accuracy Tuned DT (test set):** $(round(accuracy(cm_tuned), digits=4))"

# ╔═╡ 5b76d562-5472-416b-b63c-1fef7c81cd5f
#md"**Precision Tuned DT (test set):** $(round(precision(cm_tuned), digits=4))"

# ╔═╡ 58ee4d37-9fa9-4f09-9d37-15d9f69d5ac2
#md"**Recall Tuned DT (test set):** $(round(recall(cm_tuned), digits=4))"

# ╔═╡ 009f7612-9336-4afb-9907-4d40934e2845
#precision(cm_tuned)

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
					Child(name, Slider(1:20 , show_value = true, default=10))
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
        max_depth         = 3,
        min_samples_leaf  = trees_param.min_samples_leaf,
        min_samples_split = trees_param.min_samples_split,
        n_trees           = 3
    )

    mach_rf = machine(forest, X_train, y_train)
    fit!(mach_rf, verbosity=0)

    y_prob_rf = MLJ.predict(mach_rf, X_test)
    y_pred_rf = mode.(y_prob_rf)
    cm_rf     = confusion_matrix(y_pred_rf, y_test)
end

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
 Random Forest | $(round(accuracy(cm_rf), digits=4)) | $(round(precision(cm_rf), digits=4)) | $(round(recall(cm_rf), digits=4)) |
"""

# ╔═╡ 6fa4955d-c170-4e43-8cf6-0158cc08f60a
#begin
#    df_max_depth_range       = range(model, :max_depth,         lower=2, upper=4)
#    df_min_samples_leaf_range  = range(model, :min_samples_leaf,  lower=1, upper=2)
#    df_min_samples_split_range = range(model, :min_samples_split, lower=2, upper=4)
#
#    tuned_forest = TunedModel(
#        model      = MLJDecisionTreeInterface.RandomForestClassifier(),
#        resampling = CV(nfolds=3),          # 3 fold, niente stratified
#        range      = [df_max_depth_range, df_min_samples_leaf_range, #df_min_samples_split_range],
#        measure    = accuracy,
#        tuning     = RandomSearch(),
#        n          = 8                      # solo 8 combinazioni campionate
#    )
#
#    df_mach_tuned = machine(tuned_forest, X, y)
#    fit!(df_mach_tuned, verbosity=0)
#
#    df_y_prob_tuned = MLJ.predict(df_mach_tuned, X_test)
#    df_y_pred_tuned = mode.(df_y_prob_tuned)   # era mode.(y_prob_tuned), mancava il #prefisso df_
#    df_cm_tuned     = confusion_matrix(df_y_pred_tuned, y_test)  # stesso bug qui
#end

# ╔═╡ 524f77c3-ebe2-4e7d-bd00-538c3de83cd0
begin
	fndt = MLJ.report(mach_dt).features
	cndt = sort(MLJ.report(mach_dt).classes_seen)
	soledt = SoleModels.solemodel(fitted_params(mach_dt).tree;featurenames=fndt, classlabels = cndt)
end

# ╔═╡ d8b150a0-261a-47fc-8ad5-c071f57c2077
#begin
#    fntdt = MLJ.report(mach_tuned).best_report.features
#    cntdt = sort(MLJ.report(mach_tuned).best_report.classes_seen)
#    soletuneddt = SoleModels.solemodel(
#        fitted_params(mach_tuned).best_fitted_params.tree;
#        featurenames = fntdt,
#        classlabels = cntdt
#    )
#end

# ╔═╡ 8692525e-a66d-4413-8722-80b9f1bf436a
begin
    fnrf = MLJ.report(mach_rf).features
    cnrf = sort(MLJ.report(mach_dt).classes_seen)  # dal DT che già funziona
    solerf = SoleModels.solemodel(
        fitted_params(mach_rf).forest;
        featurenames = fnrf,
        classlabels = cnrf
    )
end

# ╔═╡ 60dda6b7-7efd-4f90-b96f-a20cce9441bc
begin
	intrees_extractor = InTreesRuleExtractor(min_coverage=1.0)
	lumen_extractor = LumenRuleExtractor()
	batrees_extractor = BATreesRuleExtractor()
	refne_extractor = REFNERuleExtractor()
	trepan_extractor = TREPANRuleExtractor()
end

# ╔═╡ cdaf88f6-d1a6-4a77-a9f6-c68eca79d364
md"""
# Model compression
TODO: with orca
"""

# ╔═╡ 69a04f86-9c42-49b1-92e3-02aaed3fd97b
md"""
# Model explanation
"""

# ╔═╡ 3cb3502b-4846-480e-acad-52ccfcac0e84
md"""
prepariamo i dati:
"""

# ╔═╡ 7008b6f7-806c-4a13-8010-8f8d1537b258
begin 
	# FEATURES
	X_train_mat = Matrix(X_train)
	X_test_mat  = Matrix(X_test)
	
	# LABELS
	y_train_vec = Vector(y_train[:, 1])
	y_test_vec  = Vector(y_test[:, 1])
end

# ╔═╡ d3f9cebb-2e31-4232-8577-66dcab631a19
md"""
Proviamo alcuni dei nostri numerosi estrattori
"""

# ╔═╡ 7ebaad1b-0f94-4649-8630-f5890bff2fed
begin
	extracted_rules_w_lumen = RuleExtraction.extractrules(lumen_extractor,solerf;minimization_scheme=:abc)
	extracted_rules_w_lumen
end

# ╔═╡ 8d28f8b0-1165-4ed2-bd3e-a09741363e83
extracted_rules_w_t = RuleExtraction.extractrules(trepan_extractor, solerf, X_test_mat)

# ╔═╡ 1395bfbb-b447-4d68-9171-35b154c4db6f
begin
    X_train_mat2 = Matrix(X_train)
    X_test_mat2 = Matrix(X_test)
    y_train_vec2 = CategoricalArrays.unwrap.(y_train)
    y_test_vec2 = CategoricalArrays.unwrap.(y_test)
end

# ╔═╡ 866f833d-dfc5-43a0-8dd5-a58679115f2b
    extracted_rules_w_intrees = RuleExtraction.extractrules(
        intrees_extractor,
        solerf,
        DataFrame(X_test),
        y_test_vec
    )

# ╔═╡ Cell order:
# ╠═3fb5d980-1477-4a63-b691-03e2b8f9e5da
# ╟─34bafc6f-ac2a-4cdb-b9c2-f766111251cb
# ╟─a1000000-3353-11f1-90b2-21952756a80b
# ╠═53c022c4-b0f3-42c0-94b0-7114bec855e7
# ╠═9e1b6b33-65e7-4eb2-a26b-b622546a2d75
# ╠═8afd5b7b-d0d7-4dcb-8911-f0fe02bd07e4
# ╟─d7df4cf0-e938-471a-8284-e741588cf830
# ╠═a2000000-3353-11f1-90b2-21952756a80b
# ╠═a3000000-3353-11f1-90b2-21952756a80b
# ╟─cfc1ddfb-3171-41df-a319-7e55b6ac79ad
# ╟─f0753df6-d698-4649-b5fd-ac7dcb385ce8
# ╠═d3fdbf72-79ee-4712-8469-4d29768b4559
# ╟─43815971-fda4-4b32-abd6-77ee7df1e1d6
# ╠═1295b232-f44e-415e-b69b-b6f20786da58
# ╠═9f0ae520-5f94-4e34-bb03-f8c68a61157a
# ╠═7d268051-0758-4c20-ae25-253a2a4627e8
# ╟─d4e2b317-92b9-4f59-b63d-91d14d3af828
# ╟─2ac0e4ca-7e77-47b8-b1b9-9e1ed0d1c426
# ╠═a6000000-3353-11f1-90b2-21952756a80b
# ╟─66ec88be-2014-40a1-a221-e8fbed52c3b7
# ╠═d0b073f2-2bba-433d-afc4-c5cc085ada62
# ╟─cdcbdbae-6867-453f-bbae-c7490a2c3df2
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
# ╠═d8b09a29-2f37-4ff5-af0f-b7c03258c8af
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
# ╠═c7000000-3353-11f1-90b2-21952756a80b
# ╠═ca74043d-78ae-47a1-a58a-a8d349313fda
# ╠═f147f0f9-5e64-4413-9142-c0ec6d081506
# ╠═bf81a25b-cbbe-4f61-8fe2-d558f13aeb5d
# ╠═6fa4955d-c170-4e43-8cf6-0158cc08f60a
# ╠═524f77c3-ebe2-4e7d-bd00-538c3de83cd0
# ╠═d8b150a0-261a-47fc-8ad5-c071f57c2077
# ╠═8692525e-a66d-4413-8722-80b9f1bf436a
# ╠═60dda6b7-7efd-4f90-b96f-a20cce9441bc
# ╠═cdaf88f6-d1a6-4a77-a9f6-c68eca79d364
# ╠═69a04f86-9c42-49b1-92e3-02aaed3fd97b
# ╠═3cb3502b-4846-480e-acad-52ccfcac0e84
# ╠═7008b6f7-806c-4a13-8010-8f8d1537b258
# ╠═d3f9cebb-2e31-4232-8577-66dcab631a19
# ╠═7ebaad1b-0f94-4649-8630-f5890bff2fed
# ╠═8d28f8b0-1165-4ed2-bd3e-a09741363e83
# ╠═1395bfbb-b447-4d68-9171-35b154c4db6f
# ╠═866f833d-dfc5-43a0-8dd5-a58679115f2b
