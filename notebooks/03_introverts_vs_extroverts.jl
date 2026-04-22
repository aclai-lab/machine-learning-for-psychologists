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

# ╔═╡ 7fe86fee-3d95-11f1-9cdb-ef83e9a88360
begin
	import Pkg
    Pkg.activate(Base.current_project(@__DIR__))
	
	using CSV
	using DataFrames
	using Statistics
	using StatsBase
	using CategoricalArrays
	
	using MLJ
	using MLJBase
		
	using Plots

	using PlutoUI
end

# ╔═╡ 1ac7f30c-3694-464a-ac25-a870bc09659b
md"""
# Introverts vs Extroverts

This notebook propose a simple data analysis and machine learning pipeline for the data provided [in this Kaggle entry](https://www.kaggle.com/datasets/yamqwe/introversionextraversion-scales/data), which in turn refers to the [openpsychometrics](https://openpsychometrics.org/) web platform, which was leveraged for collecting the data.

You can find more information in the general description on Kaggle.
"""

# ╔═╡ 1c9a69b1-2dea-42f3-bd3c-0fd33f43e475
RNG_SEED = 1605

# ╔═╡ 070d2779-dc83-4b6e-8178-07ba08d946e1
data_path = joinpath(@__DIR__, "..", "datasets", "extra", "introverts_vs_extroverts.csv")

# ╔═╡ 651e1768-c89b-40e0-bba1-24f1403f54ee
df = CSV.read(data_path, DataFrame)

# ╔═╡ b298085f-10f3-4077-b31c-ff98b2239d3c
describe(df)

# ╔═╡ e2121707-5e59-4aa2-a0d4-01dee90c1a2f
md"""
We can probably remove the questions related to age, native language and age.
"""

# ╔═╡ 550b017c-2862-4dab-b405-18781ccac1f5
df_nocommon = DataFrames.select(df, Not("gender", "engnat", "age"));

# ╔═╡ c0bf090d-c8f2-4c25-9551-356b7fda7f11
md"""
Some answering times are quite long, but the test was taken online!
"""

# ╔═╡ 3f846444-613b-40fb-9c4c-ece252813092
maximum(df_nocommon.dateload)

# ╔═╡ 5e6f3080-9cb4-41d9-aebe-e94bad3a9b1c
maximum(df_nocommon.introelapse)

# ╔═╡ 660382b4-e1f5-4f8e-b28e-2d4804c906e5
maximum(df_nocommon.testelapse)

# ╔═╡ 16e902fc-112b-4362-8117-05b431b4dc0e
maximum(df_nocommon.surveyelapse)

# ╔═╡ 65b7acfb-add1-4c4d-87ae-abbdf27648bc
df_notimes = DataFrames.select(df_nocommon, Not("dateload", "introelapse", "testelapse", "surveyelapse"));

# ╔═╡ 70757aa7-160b-4a0b-9456-eae968b9dd52
md"""
The same consideration as above can be applied to the time measurement of each question.

At this point, let us just consider the questions as variables, and let us see what we can learn from them.
"""

# ╔═╡ d8fea76b-2a92-44f9-be34-cc3fdf5bfb44
answer_cols = ["Q$(i)A" for i in 1:91]

# ╔═╡ 78a03609-4e1f-44b0-8bfb-e8024b16ad2b
df_answers = df[:, answer_cols];

# ╔═╡ 23911ccc-790b-4ec1-9511-67f07ddb611f
@bind question Select(answer_cols)

# ╔═╡ a37f2667-e18c-452d-96f9-17663b1c7e36
question_countmap = countmap(df_answers[:, question])

# ╔═╡ 38c09fd7-2181-42eb-ba03-997fbe717be9
# extract all the values within question_countmap, from the 1st key to the 5th
question_answers = reverse(collect(values(question_countmap)))

# ╔═╡ afb5d9c4-4bbd-4660-8ec8-16a3fe182948
n_answers = sum(question_answers)

# ╔═╡ bef75474-3252-496b-84ff-3f082f16763b
question_answers_percentage = question_answers ./ n_answers

# ╔═╡ e300806a-bcce-45aa-b3be-06956633c60b
pie([1,2,3,4,5], question_answers_percentage, title=question)

# ╔═╡ 696b5acd-3211-4c8f-838e-b792c3fc464c
# TODO: split df_answers in X and y

# ╔═╡ 3b620337-b77d-4664-ae7e-212b51184c48
begin
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
end

# ╔═╡ 4e15a022-62d3-428b-80c9-e278b92b7481
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

# ╔═╡ 7f136a9d-ce31-45ac-8027-701b5b48f7cb


# ╔═╡ Cell order:
# ╟─1ac7f30c-3694-464a-ac25-a870bc09659b
# ╠═7fe86fee-3d95-11f1-9cdb-ef83e9a88360
# ╠═1c9a69b1-2dea-42f3-bd3c-0fd33f43e475
# ╠═070d2779-dc83-4b6e-8178-07ba08d946e1
# ╠═651e1768-c89b-40e0-bba1-24f1403f54ee
# ╠═b298085f-10f3-4077-b31c-ff98b2239d3c
# ╟─e2121707-5e59-4aa2-a0d4-01dee90c1a2f
# ╠═550b017c-2862-4dab-b405-18781ccac1f5
# ╟─c0bf090d-c8f2-4c25-9551-356b7fda7f11
# ╠═3f846444-613b-40fb-9c4c-ece252813092
# ╠═5e6f3080-9cb4-41d9-aebe-e94bad3a9b1c
# ╠═660382b4-e1f5-4f8e-b28e-2d4804c906e5
# ╠═16e902fc-112b-4362-8117-05b431b4dc0e
# ╠═65b7acfb-add1-4c4d-87ae-abbdf27648bc
# ╟─70757aa7-160b-4a0b-9456-eae968b9dd52
# ╠═d8fea76b-2a92-44f9-be34-cc3fdf5bfb44
# ╠═78a03609-4e1f-44b0-8bfb-e8024b16ad2b
# ╠═23911ccc-790b-4ec1-9511-67f07ddb611f
# ╠═a37f2667-e18c-452d-96f9-17663b1c7e36
# ╠═38c09fd7-2181-42eb-ba03-997fbe717be9
# ╠═afb5d9c4-4bbd-4660-8ec8-16a3fe182948
# ╠═bef75474-3252-496b-84ff-3f082f16763b
# ╠═e300806a-bcce-45aa-b3be-06956633c60b
# ╠═696b5acd-3211-4c8f-838e-b792c3fc464c
# ╠═3b620337-b77d-4664-ae7e-212b51184c48
# ╠═4e15a022-62d3-428b-80c9-e278b92b7481
# ╠═7f136a9d-ce31-45ac-8027-701b5b48f7cb
