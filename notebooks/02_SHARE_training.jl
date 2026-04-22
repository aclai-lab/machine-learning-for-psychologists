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
	using CategoricalArrays
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

It is very common to see "doubly-connected channels" between MLJ and other packages of the Julia community! As we shall later, Sole is no exception, and the two frameworks can be used in synergy.
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

# ╔═╡ 3f185c7c-cbc8-4d33-be2c-57d72784c51e
md"""
# One-hot encoding

Multiclass labels are typically stored as categorical values, and the model must be able to interpret them correctly as distinct classes rather than arbitrary text or numbers.

It is by no means certain that a model can naturally handle Multiclass.

A safe, general solution is... one-hot encoding!

It is a way of representing each category as a separate binary feature, where only one feature is "active" (i.e., set to 1) for a given observation and all others are 0.

With one-hot encoding:
- we avoid introducing a false notion of ordering between the values of an attribute (as in the case of leveraging Multiclass types);
- it is not mandatory for the trained model to be designed for handling categorical attributes.

"""

# ╔═╡ d0b073f2-2bba-433d-afc4-c5cc085ada62
LocalResource("../images/onehot_encoding.png")

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
X_ninstances, X_nattr+ibutes = size(X)

# ╔═╡ b0000000-3353-11f1-90b2-21952756a80b
begin
    (X_train, X_test), (y_train, y_test) = partition(
        (X, y), 0.7;

		# sometimes, data is ordered via a criterion we do not want to assume
		shuffle=true,

		# we need this to keep Xs and ys glued pairwise
		multi=true,

		# for reproducibility
		rng=RNG_SEED
    )
end

# ╔═╡ 0e734fdc-969b-46c6-a9c9-fca950afd09a
size(X_train)

# ╔═╡ 54e24bc2-3fe6-4c6b-acf2-96de70e2736b


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

Actually, we define 10 different trees with different settings of their *hyperparameters*; then, we select one specific tree with a slider and proceed to train it.

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

# ╔═╡ 99f3b672-b24e-4d09-a1bb-cc4b2f928347
md"""
The two commands below demonstrate that the specific implementation of decision tree coming from `DecisionTree.jl` is not compatible with `Multiclass` scientific type (whilst the one coming from `BetaML` could be fine by-design!).
"""

# ╔═╡ a1ee0585-0fb7-4019-ae32-fc2fbf9588b3
MLJ.models(matching(X_raw, y))

# ╔═╡ c9999307-8118-4b39-b74f-8296685035b6
for model in MLJ.models(matching(X_raw, y))
	if model.name == "DecisionTreeClassifier"
		println(model)
	end
end

# ╔═╡ d8b09a29-2f37-4ff5-af0f-b7c03258c8af
MLJ.models(matching(X_train, y_train))

# ╔═╡ 3c57b20b-42c0-464e-be47-ce653dfff359
for model in MLJ.models(matching(X_train, y_train))
	if model.name == "DecisionTreeClassifier"
		println(model)
	end
end

# ╔═╡ 125b244f-762a-449a-ac71-daa428650f81
md"""
!!! info "Exercise"
	Select the type `MLJDecisionTreeInterface.DecisionTreeClassifier` and click the "Live docs" in the bottom-right button.

	We want to answer two questions:
	1. how is it called the specific algorithm implemented by `DecisionTrees.jl` for inducing decision tree models?
	2. which hyperparameters are available?
"""

# ╔═╡ df689893-a895-4b3e-85a2-5364274bf575
model = MLJDecisionTreeInterface.DecisionTreeClassifier(
	max_depth = 10,
	min_samples_leaf = 1,
	min_samples_split = 2,
	min_purity_increase = 0.0,
	n_subfeatures = 0.0,
	post_prune = false,
	merge_purity_threshold = 0.9,
	rng = RNG_SEED
)

# ╔═╡ b5000000-3353-11f1-90b2-21952756a80b
begin
    mach_dt = machine(model, X_train, y_train)
    fit!(mach_dt)
    y_prob_dt = MLJ.predict(mach_dt, X_test)
    y_pred_dt = mode.(y_prob_dt)
    cm_dt = confusion_matrix(y_pred_dt, y_test)
end

# ╔═╡ 1f8b37ed-336b-4f6e-9d47-643bdf5295d7
md"""
To assess the quality of the trained model, we leverage a *confusion matrix* (see below).

In general, we are interested in three metrics: *accuracy*, *precision* and *recall* (or *sensitivity*, as indicated below).

**Accuracy** tells us how often the model is right across all classes; it is about the overall proportion of correct predictions but **can be misleading when data is unbalanced**.

**Precision** is the proportion of predicted positives that are actually correct; this indicates "how reliable" positive predictions are.

**Recall** is the proportion of actual positives that are captured; if the recall is low, then just a few positives are captured.
"""

# ╔═╡ c8ac4328-8f16-4033-baec-2b7861c4010f
LocalResource("../images/confusion_matrix.png")

# ╔═╡ ac27ff8d-dacb-4cae-b7e3-cc70135feaa6
md"""
!!! info "Exercise"
	What happens when the recall is low and the precision is high?

	What happens when the converse holds?
"""

# ╔═╡ b6000000-3353-11f1-90b2-21952756a80b
cm_dt

# ╔═╡ b7000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Decision Tree (test set):** $(round(accuracy(cm_dt), digits=4))"

# ╔═╡ edc4f8f8-12b9-45b2-bbbe-32736dc6fbd3
md"**Precision Decision Tree (test set):** $(round(precision(cm_dt), digits=4))"

# ╔═╡ 948beef6-c940-4852-9500-76fb521aa437
md"**Recall Decision Tree (test set):** $(round(recall(cm_dt), digits=4))"

# ╔═╡ d5458851-5414-484c-82cc-4e63b5ec06ef
md"""
!!! warning "Cross validation"
	Beware: the performance of our trained model could depend on the partitioning of data in the training and test set.

	To remove luck from the process we can rely on a more robust *training schema*, called *cross validation*.

	Essentially, **cross validation** is about splitting the original data into multiple folds and repeatedly training and testing the model on different folds.

	An important variant, in the context of this work, is **stratified cross validation**, in which each fold preserves the class distribution of the original dataset; this guarantees a stable evaluation of the model (there are )
	Stratified cross validation  follows the same procedure but ensures that each fold preserves the class distribution of the original dataset.
"""

# ╔═╡ d3227330-0101-4253-838e-6f916fbbd18c
LocalResource("../images/cross_validation.png")

# ╔═╡ fd2a40f9-20f1-44a7-8231-9796dffd0922
LocalResource("../images/stratified_cross_validation.png")

# ╔═╡ b8000000-3353-11f1-90b2-21952756a80b
begin
    mach_cv = machine(model, X, y)
    acc_cv = evaluate!(
        mach_cv;
        resampling = StratifiedCV(nfolds=10, shuffle=true, rng=RNG_SEED),
        measures   = [accuracy],
        verbosity  = 0
    )
    acc_cv
end

# ╔═╡ b9000000-3353-11f1-90b2-21952756a80b
md"""
# Hyperparameter tuning

To avoid arbitrary settings of hyperparameters, we can rely on a systematic procedure called *grid search* (i.e., we try all the possible combinations considering many domains).
"""

# ╔═╡ c0000000-3353-11f1-90b2-21952756a80b
 begin
     max_depth_range         = range(model, :max_depth,          lower=2, upper=10)
     min_samples_leaf_range  = range(model, :min_samples_leaf,   lower=1, upper=5)
     min_samples_split_range = range(model, :min_samples_split,  lower=2, upper=10)
 
     tuned_tree = TunedModel(
		model      = MLJDecisionTreeInterface.DecisionTreeClassifier(),
        resampling = StratifiedCV(nfolds=10, shuffle=true),
        range      = [
			max_depth_range, 
			min_samples_leaf_range, 
			min_samples_split_range
		 ],
        measure    = accuracy,
        tuning     = RandomSearch()
     )
 
     mach_tuned = machine(tuned_tree, X, y)
     fit!(mach_tuned, verbosity=0)
 
     y_prob_tuned = MLJ.predict(mach_tuned, X_test)
     y_pred_tuned = mode.(y_prob_tuned)
     cm_tuned     = confusion_matrix(y_pred_tuned, y_test)
 end

# ╔═╡ c2000000-3353-11f1-90b2-21952756a80b
md"**Accuracy Tuned DT (test set):** $(round(accuracy(cm_tuned), digits=4))"

# ╔═╡ 5b76d562-5472-416b-b63c-1fef7c81cd5f
md"**Precision Tuned DT (test set):** $(round(precision(cm_tuned), digits=4))"

# ╔═╡ 58ee4d37-9fa9-4f09-9d37-15d9f69d5ac2
md"**Recall Tuned DT (test set):** $(round(recall(cm_tuned), digits=4))"

# ╔═╡ 98213623-eaec-4928-bfea-c0f0c4f04f90
md"""
!!! tip "☀️ Model inspection with Sole ☀️"
	Try to describe a typical path in the decision tree below.

	Now it may look a little bit cumbersome, but we are going to definitely simplify this theory in a moment, leveraging Sole!
"""

# ╔═╡ 5117a6c9-b090-4d85-9cbc-9500beb09f7e
SoleModels.solemodel(fitted_params(mach_dt).tree)

# ╔═╡ c3000000-3353-11f1-90b2-21952756a80b
md"## Random Forest"

# ╔═╡ 2b686c03-03df-41c4-a78f-71aae61ad5ff
LocalResource("../images/random_forest.png")

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
| Decision Tree (depth=10) | $(round(accuracy(cm_dt), digits=4)) | $(round(precision(cm_dt), digits=4)) | $(round(recall(cm_dt), digits=4)) |
| Decision Tree Tuned | $(round(accuracy(cm_tuned), digits=4)) | $(round(precision(cm_tuned), digits=4)) | $(round(recall(cm_tuned), digits=4)) |
| Random Forest | $(round(accuracy(cm_rf), digits=4)) | $(round(precision(cm_rf), digits=4)) | $(round(recall(cm_rf), digits=4)) |
"""

# ╔═╡ 6fa4955d-c170-4e43-8cf6-0158cc08f60a
begin
    df_max_depth_range       = range(model, :max_depth,         lower=2, upper=4)
    df_min_samples_leaf_range  = range(model, :min_samples_leaf,  lower=1, upper=2)
    df_min_samples_split_range = range(model, :min_samples_split, lower=2, upper=4)

    tuned_forest = TunedModel(
        model      = MLJDecisionTreeInterface.RandomForestClassifier(),
        resampling = CV(nfolds=3), 			# low, for keeping the computation light
        range      = [df_max_depth_range, df_min_samples_leaf_range, df_min_samples_split_range],
        measure    = accuracy,
        tuning     = RandomSearch(),
        n          = 8                      # low, for keeping the computation light
    )

    df_mach_tuned = machine(tuned_forest, X, y)
    fit!(df_mach_tuned, verbosity=0)

    df_y_prob_tuned = MLJ.predict(df_mach_tuned, X_test)
    df_y_pred_tuned = mode.(df_y_prob_tuned)
    df_cm_tuned     = confusion_matrix(df_y_pred_tuned, y_test)
end

# ╔═╡ 524f77c3-ebe2-4e7d-bd00-538c3de83cd0
begin
	featurenames_decisiontree = MLJ.report(mach_dt).features
	classnames_decisiontree = sort(MLJ.report(mach_dt).classes_seen)
	
	sole_decisiontree =  SoleModels.solemodel(
		fitted_params(mach_dt).tree;
		featurenames = featurenames_decisiontree, 
		classlabels = classnames_decisiontree
	)
end

# ╔═╡ d8b150a0-261a-47fc-8ad5-c071f57c2077
begin
    featurenames_tunedtree = MLJ.report(mach_tuned).best_report.features
    classnames_tunedtree = sort(MLJ.report(mach_tuned).best_report.classes_seen)
	
    sole_tunedtree = SoleModels.solemodel(
        fitted_params(mach_tuned).best_fitted_params.tree;
        featurenames = featurenames_tunedtree,
        classlabels = classnames_tunedtree
    )
end

# ╔═╡ 8692525e-a66d-4413-8722-80b9f1bf436a
begin
    featurenames_randomforest = MLJ.report(mach_rf).features
    classnames_randomforest = sort(MLJ.report(mach_dt).classes_seen) 
	
    sole_randomforest = SoleModels.solemodel(
        fitted_params(mach_rf).forest;
        featurenames = featurenames_randomforest,
        classlabels = classnames_randomforest
    )
end

# ╔═╡ 9319d70b-e1ae-493f-92bf-42745840411a
md"""
# PostHoc Analysis

Now that we trained three variations of decision trees from our data, we are interested in *compress their representation* and *extract the essential rules* from them.
"""

# ╔═╡ 7008b6f7-806c-4a13-8010-8f8d1537b258
begin 
	# from now onwards, it is better to interpret our DataFrames as Matrixes and Vectors
	X_train_mat = Matrix(X_train);
	X_test_mat  = Matrix(X_test);

    X_train_mat_f = Matrix{Float64}(X_train);
    X_test_mat_f  = Matrix{Float64}(X_test);
		
    y_train_str = string.(y_train[:, 1]);
    y_test_str  = string.(y_test[:, 1]);
end

# ╔═╡ cdaf88f6-d1a6-4a77-a9f6-c68eca79d364
md"""
# Model compression

Once a model has been trained, it can be quite large and complex, especially in the case of a random forest, which is an ensemble of many trees.

While this complexity contributes to predictive accuracy, it can make the model hard to interpret and computationally expensive to consult in practice.

To mitigate the problem, we can exploit **model compression** for producing new, smaller models that approximate the behavior of the original one as closely as possible, while being significantly simpler in structure.

In particular, here we leverage the novel **Orca** (Optimized aRbitrary-ensemble Compression Algorithm), which is based on [evolutionary optimization](https://en.wikipedia.org/wiki/Evolutionary_algorithm). 

!!! tip "The three complexity dimensions"
	Orca optimizes over three *complexity dimensions*:

	- size: the total number of nodes in a tree;
	- depth: the maximum number of conditions that must be checked to reach a prediction;
	- dimensionality: the number of unique variables leveraged across all splits.

	These three dimensions can be optimized both individually and in combinations.
"""

# ╔═╡ 9955fa39-3aae-4cf8-81e6-a2e3ce7d5615
# with :size_depth we ask to minimize both size and depth simultaneously 
compressed_size_depth = SolePostHoc.Orca.compression(
    sole_randomforest, :size_depth, X_train_mat_f, y_train_str;
    population_size=3, n_generations=3
)

# ╔═╡ b7b72708-4f7a-4a5b-a898-b0487b4ef44e
compressed_only_depth = SolePostHoc.Orca.compression(
    sole_randomforest, :depth, X_train_mat_f, y_train_str;
    population_size=3, n_generations=3
)

# ╔═╡ 7f731578-78c6-4c06-8cf8-a121fb7c0345
# with :full_dimensional we take into account all the three complexity dimensions
compressed_full_dimensional = SolePostHoc.Orca.compression(
    sole_randomforest, :full_dimensional, X_train_mat_f, y_train_str;
    population_size=3, n_generations=4
)

# ╔═╡ 69a04f86-9c42-49b1-92e3-02aaed3fd97b
md"""
# Model explanation

Although a model like a random forest can potentially be interpretable (i.e., easy to explain, discuss and understand), it often reveals itself as a *black box*, which is very hard for humans to read.

Our goal, here, is to produce human-readable and highly descriptive sets of decisions which are characteristics of the whole forest.

!!! success "Rule extractors in SolePostHoc"
	| Extractor | Strategy |
	|-----------|----------|
	| `InTreesRuleExtractor` | Enumerates paths directly from the trees in the forest |
	| `LumenRuleExtractor` | Optimizes rules for *coverage* and *compactness* |
	| `BATreesRuleExtractor` | Builds an approximating single tree from the forest |
	| `REFNERuleExtractor` | Refines rules by focusing on misclassified instances |
	| `TREPANRuleExtractor` | Queries the forest as an oracle to induce a new tree |


"""

# ╔═╡ 5723b1e8-b594-4893-b71d-3df41fe49393
md"""
These are some of the rule extractors available via Sole.jl.
"""

# ╔═╡ 0946aa01-90cf-4747-915b-f6806995a36b
begin
	intrees_extractor = InTreesRuleExtractor(min_coverage=1.0)
	lumen_extractor = LumenRuleExtractor()
	batrees_extractor = BATreesRuleExtractor()
	refne_extractor = REFNERuleExtractor()
	trepan_extractor = TREPANRuleExtractor()
end

# ╔═╡ c4aa241b-d4da-4b08-9143-9ff2754564cc
md"""
### Start the 💡 Lumen extractor 💡
"""

# ╔═╡ fa4f5fc2-aa91-484a-9544-f09a36857db1
@bind start_lumen_extractor Switch(; default=false)

# ╔═╡ 7ebaad1b-0f94-4649-8630-f5890bff2fed
if start_lumen_extractor
	extracted_rules_w_lumen = RuleExtraction.extractrules(
		lumen_extractor,
		sole_randomforest;
		minimization_scheme=:abc
	)
	
	extracted_rules_w_lumen
end

# ╔═╡ 3f87b4ca-a1ca-4622-8ed6-edc18480dc63
md"""
### Start Trepan extractor
"""

# ╔═╡ 1d3eb7ab-3765-485e-bfdd-0ca236004e75
@bind start_trepan_extractor Switch(; default=false)

# ╔═╡ 8d28f8b0-1165-4ed2-bd3e-a09741363e83
if start_trepan_extractor
	extracted_rules_w_t = RuleExtraction.extractrules(
		trepan_extractor, 
		sole_randomforest, 
		X_test_mat
	)
end

# ╔═╡ f93ecd9e-3cad-4afd-a6e3-fbabf9f347a5
md"""
### Start Intrees extractor
"""

# ╔═╡ bfdb2bfd-8c23-4f04-b60e-b60ff5a927fd
@bind start_intrees_extractor Switch(; default=false)

# ╔═╡ 866f833d-dfc5-43a0-8dd5-a58679115f2b
if start_intrees_extractor
	extracted_rules_w_intrees = RuleExtraction.extractrules(
        intrees_extractor,
        sole_randomforest,
        DataFrame(X_test),
        y_test_vec
    )
end

# ╔═╡ eb0c334f-2be9-45ca-b4d6-b747396d9a6d
md"""
# Rule explanation 
"""

# ╔═╡ b48b9625-22bf-4d64-bf88-5c6d923e196d
try
	pretty_print_decision_set(extracted_rules_w_lumen)
catch e
	if e isa UndefVarError
		print("You first need to execute the cell 'Start the 💡 Lumen extractor 💡'")
	end
end


# ╔═╡ bbfdc551-476d-4d28-b676-f5477fc8c12b
md"""
# Challenging the LLMs
In the cells below, we repeat the same prompting exercise as in the first notebook, asking ChatGPT to clarify some hypothetical doubts related to this lesson.
"""

# ╔═╡ af48c01f-a98d-43c2-9a14-d4c780f86590
md"""
### Prompt #1

This is extremely naïve.

```
Consider this variables, coming from a dataset I am studying: Suggest me a machine learning training pipeline for a dataset having these header: age_int,hhsize,dn042_,dn503_,dn014_,dn034_,iv009_,hh022_,hh025_,hh017e,sp002_,sp008_,ch001_,ch021_,ep005_,co007_,ac035d1,ac035d4,ac035d5,ac035d7,ac035d8,ac035d9,ac035d10,ac035dno,ac012_,ac014_,ac015_,ac016_,ac017_,ac018_,ac019_,ac020_,ac021_,ac022_,ac023_,ac024_,ac025_,it003_,euro1,euro2,euro3,euro4,euro5,euro6,euro7,euro8,euro9,euro10,euro11,euro12,bmi2,phactiv,country,language,ph003_,ph004_,ph005_,ph006d1,ph006d2,ph006d3,ph006d4,ph006d5,ph006d6,ph006d10,ph006d11,ph006d12,ph006d13,ph006d14,ph006d15,ph006d16,ph006d18,ph006d19,ph006d20,ph006dno,ph006dot,ph008d2,ph008d3,ph008d4,ph008d5,ph008d6,ph008d7,ph008d8,ph008d9,ph008d10,ph008d11,ph008d12,ph008d13,ph008d14,ph008d15,ph008d16,ph008d17,ph008d18,ph008d19,ph008d20,ph008d21,ph008d22,ph008dot,ph011d1,ph011d2,ph011d3,ph011d4,ph011d6,ph011d7,ph011d8,ph011d9,ph011d10,ph011d11,ph011d13,ph011d14,ph011d15,ph011dno,ph011dot,ph012_,ph013_,ph041_,ph043_,ph044_,ph045_,ph046_,ph048d1,ph048d2,ph048d3,ph048d4,ph048d5,ph048d6,ph048d7,ph048d8,ph048d9,ph048d10,ph048dno,ph049d1,ph049d2,ph049d3,ph049d4,ph049d5,ph049d6,ph049d7,ph049d8,ph049d9,ph049d10,ph049d11,ph049d12,ph049d13,ph049dno,ph061_,ph065_,ph066_,ph071_1,ph071_2,ph071_3,ph071_4,ph072_1,ph072_2,ph072_3,ph072_4,ph073_1,ph073_2,ph073_3,ph073_4,ph074_1,ph074_2,ph074_3,ph074_4,ph075_1,ph075_2,ph075_3,ph075_4,ph076_1,ph076_2,ph076_3,ph076_4,ph077_1,ph077_2,ph077_3,ph077_4,ph084_,ph085_,ph089d1,ph089d2,ph089d3,ph089d4,ph090_,ph091_,ph092_,ph094_,ph095_,mh037_,br015_,br016_,isced1997_r,initial_euro_d,euro_d --- How can i decide whether an instance is depressed or not? For example, consider this instance and find me the value for "???" 61.0,1.0,2.0,,,,4.0,1.0,1.0,15000.0,5.0,5.0,2.0,3.0,1.0,4.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,10.0,3.0,3.0,4.0,1.0,4.0,4.0,1.0,1.0,1.0,1.0,2.0,1.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0,0.0,11.0,11.0,3.0,5.0,3.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,70.0,166.0,5.0,2.0,2.0,5.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,5.0,5.0,,,,,,5.0,5.0,5.0,5.0,,,,,,,,,,,,,,,,,,,,,5.0,,0.0,0.0,0.0,0.0,,5.0,7.0,2.0,,3.0,1.0,1.0,5.0,no,???
```
"""

# ╔═╡ bbf77d77-41d8-471b-84d6-77bca60ee9f5
md"""
!!! info "Exercise"
	- Do you think the prompt above can give you some insights?
	- If so, what is the price you have to pay?
"""

# ╔═╡ 70529231-d37d-4b23-bb62-d73fc3afc977
md"""
### Prompt #2
```
Give me the minimal explanation rule for this random forest:
▣ ([difficulty_none_adl__0.0] < 0.5)
├✔ ([life_satisfaction] < 7.5)
│ ├✔ ([gender__2.0] < 0.5)
│ │ ├✔ ([pain__1.0] < 0.5)
│ │ │ ├✔ ([life_satisfaction] < 5.5)
│ │ │ │ ├✔ ([feel_full_of_energy__1.0] < 0.5)
│ │ │ │ │ ├✔ no
│ │ │ │ │ └✘ no
│ │ │ │ └✘ ([out_of_control__2.0] < 0.5)
│ │ │ │   ├✔ no
│ │ │ │   └✘ no
etc...
```
"""

# ╔═╡ 34a6e1cf-ea00-4f5a-9750-8c705c4c9118
LocalResource("../images/prompt-02-02.png")

# ╔═╡ Cell order:
# ╟─34bafc6f-ac2a-4cdb-b9c2-f766111251cb
# ╠═a1000000-3353-11f1-90b2-21952756a80b
# ╠═53c022c4-b0f3-42c0-94b0-7114bec855e7
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
# ╟─3f185c7c-cbc8-4d33-be2c-57d72784c51e
# ╠═d0b073f2-2bba-433d-afc4-c5cc085ada62
# ╠═67a333d7-4aa7-45bd-ad05-957fc87102f2
# ╠═51278f4e-d554-48f6-a63f-14f3e78fee17
# ╠═cf3b8afb-f32d-42f1-aa7e-7dfc0ec0ef17
# ╠═4ccc6d8a-a6c3-42c5-bdcd-e489bcdb6798
# ╠═565c7e32-c824-41e8-a8ea-4d1e4a638c86
# ╠═a7000000-3353-11f1-90b2-21952756a80b
# ╠═b0000000-3353-11f1-90b2-21952756a80b
# ╠═0e734fdc-969b-46c6-a9c9-fca950afd09a
# ╠═54e24bc2-3fe6-4c6b-acf2-96de70e2736b
# ╟─6abe7cf4-231c-4f75-839f-6b80891d3088
# ╠═c70c1204-60a1-4ecf-b0a1-8a60938686ff
# ╠═d9bac238-70b3-43a0-95c5-99fb5ae96b92
# ╠═57d8be18-2370-4fbc-bc69-eb54898e9dff
# ╟─6b94d7db-7f2f-4471-8999-b138b6b0c448
# ╟─64d0496d-51f0-48f1-9068-f34328d7a857
# ╠═76edf51c-7ca6-49d1-85e8-bab1b77c04c2
# ╠═b1000000-3353-11f1-90b2-21952756a80b
# ╟─99f3b672-b24e-4d09-a1bb-cc4b2f928347
# ╠═a1ee0585-0fb7-4019-ae32-fc2fbf9588b3
# ╠═c9999307-8118-4b39-b74f-8296685035b6
# ╠═d8b09a29-2f37-4ff5-af0f-b7c03258c8af
# ╠═3c57b20b-42c0-464e-be47-ce653dfff359
# ╟─125b244f-762a-449a-ac71-daa428650f81
# ╟─df689893-a895-4b3e-85a2-5364274bf575
# ╠═b5000000-3353-11f1-90b2-21952756a80b
# ╟─1f8b37ed-336b-4f6e-9d47-643bdf5295d7
# ╠═c8ac4328-8f16-4033-baec-2b7861c4010f
# ╟─ac27ff8d-dacb-4cae-b7e3-cc70135feaa6
# ╠═b6000000-3353-11f1-90b2-21952756a80b
# ╟─b7000000-3353-11f1-90b2-21952756a80b
# ╟─edc4f8f8-12b9-45b2-bbbe-32736dc6fbd3
# ╟─948beef6-c940-4852-9500-76fb521aa437
# ╟─d5458851-5414-484c-82cc-4e63b5ec06ef
# ╠═d3227330-0101-4253-838e-6f916fbbd18c
# ╠═fd2a40f9-20f1-44a7-8231-9796dffd0922
# ╠═b8000000-3353-11f1-90b2-21952756a80b
# ╟─b9000000-3353-11f1-90b2-21952756a80b
# ╠═c0000000-3353-11f1-90b2-21952756a80b
# ╟─c2000000-3353-11f1-90b2-21952756a80b
# ╟─5b76d562-5472-416b-b63c-1fef7c81cd5f
# ╟─58ee4d37-9fa9-4f09-9d37-15d9f69d5ac2
# ╟─98213623-eaec-4928-bfea-c0f0c4f04f90
# ╠═5117a6c9-b090-4d85-9cbc-9500beb09f7e
# ╟─c3000000-3353-11f1-90b2-21952756a80b
# ╟─2b686c03-03df-41c4-a78f-71aae61ad5ff
# ╠═c4000000-3353-11f1-90b2-21952756a80b
# ╠═9410b8b8-10f0-460b-b46c-a7715cee1fe2
# ╠═b6bc4920-7538-4df9-8295-4730db58be77
# ╠═b03265a4-0776-41c9-846a-5e74b459814d
# ╠═1d0cc054-5c38-46d5-9033-4872d265c0d8
# ╠═c5000000-3353-11f1-90b2-21952756a80b
# ╟─c7000000-3353-11f1-90b2-21952756a80b
# ╟─ca74043d-78ae-47a1-a58a-a8d349313fda
# ╟─f147f0f9-5e64-4413-9142-c0ec6d081506
# ╠═bf81a25b-cbbe-4f61-8fe2-d558f13aeb5d
# ╠═6fa4955d-c170-4e43-8cf6-0158cc08f60a
# ╠═524f77c3-ebe2-4e7d-bd00-538c3de83cd0
# ╠═d8b150a0-261a-47fc-8ad5-c071f57c2077
# ╠═8692525e-a66d-4413-8722-80b9f1bf436a
# ╟─9319d70b-e1ae-493f-92bf-42745840411a
# ╠═7008b6f7-806c-4a13-8010-8f8d1537b258
# ╟─cdaf88f6-d1a6-4a77-a9f6-c68eca79d364
# ╠═9955fa39-3aae-4cf8-81e6-a2e3ce7d5615
# ╠═b7b72708-4f7a-4a5b-a898-b0487b4ef44e
# ╠═7f731578-78c6-4c06-8cf8-a121fb7c0345
# ╟─69a04f86-9c42-49b1-92e3-02aaed3fd97b
# ╟─5723b1e8-b594-4893-b71d-3df41fe49393
# ╠═0946aa01-90cf-4747-915b-f6806995a36b
# ╟─c4aa241b-d4da-4b08-9143-9ff2754564cc
# ╠═fa4f5fc2-aa91-484a-9544-f09a36857db1
# ╠═7ebaad1b-0f94-4649-8630-f5890bff2fed
# ╟─3f87b4ca-a1ca-4622-8ed6-edc18480dc63
# ╠═1d3eb7ab-3765-485e-bfdd-0ca236004e75
# ╠═8d28f8b0-1165-4ed2-bd3e-a09741363e83
# ╟─f93ecd9e-3cad-4afd-a6e3-fbabf9f347a5
# ╠═bfdb2bfd-8c23-4f04-b60e-b60ff5a927fd
# ╠═866f833d-dfc5-43a0-8dd5-a58679115f2b
# ╟─eb0c334f-2be9-45ca-b4d6-b747396d9a6d
# ╠═b48b9625-22bf-4d64-bf88-5c6d923e196d
# ╠═bbfdc551-476d-4d28-b676-f5477fc8c12b
# ╟─af48c01f-a98d-43c2-9a14-d4c780f86590
# ╟─bbf77d77-41d8-471b-84d6-77bca60ee9f5
# ╟─70529231-d37d-4b23-bb62-d73fc3afc977
# ╠═34a6e1cf-ea00-4f5a-9750-8c705c4c9118
