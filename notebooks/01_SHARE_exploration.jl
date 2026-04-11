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

# ╔═╡ 494947b5-219b-43ad-b29f-56216b3dc639
begin
	import Pkg
	Pkg.activate(Base.current_project(@__DIR__))

	# necessary to read and write the dataset from binary .sav files
	using FileIO
	using StatFiles
	using Serialization
	
	# libraries for supporting various data encodings
	using CSV
	using DataFrames
	using CategoricalArrays

	# general utilities for data exploration and values imputation
	using Statistics
	using StatsBase
	# using Impute

	# our own utilities for cleaning data!
	INCLUDE_PATH = joinpath(@__DIR__, "..", "utils")
	include(joinpath(INCLUDE_PATH, "filters.jl"));
	include(joinpath(INCLUDE_PATH, "adapters.jl"));

	# for the interactive Pluto's environment and plotting
	using PlutoUI
	using Plots
	using StatsPlots	
end

# ╔═╡ 49733da1-b29a-41cd-a1dd-3d748ca70f97
md"""
# Imports
"""

# ╔═╡ a6c8a233-8c6f-43b2-a4fd-a0bbc6425d74
md"""
# Data Loading

In this section, we are going to load the raw data collected by Murri et al., as described in their work [Risk Prediction Models for Depression in Community-Dwelling Older Adults](https://www.ajgponline.org/article/S1064-7481(22)00435-3/abstract). 

The data was collected with the final goal of developing machine-learning predictors for late-life depression, including demographic characteristics, health-related factors, disability and individual depressive symptoms.

In particular, here we are dealing with data from wave 5 (collected in 2013), consisting of baseline and retrospective information, and outcome data from wave 6 (collected in 2015). 
"""

# ╔═╡ e11b7142-8349-452e-9f8c-aaf006d90790
md"""
!!! tip
	In the cell below we define the *filepath* of the binary data to read, `SAV_PATH`, and the filepath where the human-readable CSV version must be saved, `CSV_PATH`.

	To do so, we leverage the `joinpath` function and the `@__DIR__` shortcut; these are crucial to guarantee the reproducibility of your data analysis!

	Try to leverage the documentation of Julia to learn about them (click *Live docs* in the bottom-right corner).
"""

# ╔═╡ e202a63d-b119-47ac-bf70-6516fb29f423
begin
	DATASET_FOLDER = joinpath(@__DIR__, "..", "datasets");
	
	DATA_PATH = joinpath(DATASET_FOLDER, "not_onco_combined_dataset");
	SAV_PATH = "$(DATA_PATH).sav";
	CSV_PATH = "$(DATA_PATH).csv";
end

# ╔═╡ f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
df_raw = DataFrame(load(SAV_PATH))

# ╔═╡ c5e327e5-a169-4f25-a09d-d0c4f6540d9d
md"""
!!! info
	[Click here](https://en.wikipedia.org/wiki/Comma-separated_values) to know more about CSV (comma-separated values).
"""

# ╔═╡ d3a4de4f-10c8-4e70-9104-b47364b99179
md"""
# Data Exploration

First of all, we want to take a bit of familiarity with data.
At the end of this section we should be able to answer the following questions:
1. how many instances and attributes are there?
2. what do attributes encode? which are categorical and which are continuous?
3. looking at the distributions, are there attributes that are trivially uninformative? (e.g., having always the same value)
"""

# ╔═╡ 7db28b02-a364-4bd0-84fe-be7c2d87e2cc
md"""
!!! info 
	To decipher the meaning of each attribute code, we need to consult the [supplementary material](https://dk.aclai.unife.it/Supplementary_information.pdf) shared with the work of Murri et al. or, more generally, the [official SHARE archive website](https://www.share-datadocutool.org/study-units/view/6).

	The mapping `attribute_names` will be useful to translate the attribute names of in `df`.
"""

# ╔═╡ c88ab2ca-0a1a-4065-b1bb-10858e45b599
attribute_names = Dict(
	# Section 1: Sociodemographic
	"age_int" => "age",
    "dn042_" => "gender",
    "dn503_" => "ethnicity",
    "dn014_" => "marital_status",
    "iv009_" => "residence_rural_urban",
    "hhsize" => "household_size",
	"isced1997_r" => "education_level",

	# social contacts
    "sp002_" => "social_support_received",
    "sp008_" => "social_support_given",
    "ch001_" => "number_of_children",
    "ch021_" => "number_of_grandchildren",
    "dn034_" => "number_of_siblings",
    "ep005_" => "occupation_employment",

	# activities
    "ac035d1" => "voluntary_charity_work",
    "ac035d4" => "educational_training_course",
    "ac035d5" => "sport_social_club",
    "ac035d7" => "political_community_org",
    "ac035d8" => "reading_books_magazines",
    "ac035d9" => "word_number_games",
    "ac035d10" => "cards_chess_games",
    "ac035dno" => "no_activities",
    "it003_" => "computer_skills",
    "hh022_" => "perception_of_neighbourhood",
    "hh025_" => "people_who_would_help",
    "hh017e" => "low_income",

	# Section 2: Mental Health
	
	# depression symptoms (EURO-D standard)
    "euro1" => "depression_symptom_depression",
    "euro2" => "depression_symptom_pessimism",
    "euro3" => "depression_symptom_suicidality",
    "euro4" => "depression_symptom_guilt",
    "euro5" => "depression_symptom_sleep",
    "euro6" => "depression_symptom_interest",
    "euro7" => "depression_symptom_irritability",
    "euro8" => "depression_symptom_appetite",
    "euro9" => "depression_symptom_fatigue",
    "euro10" => "depression_symptom_concentration",
    "euro11" => "depression_symptom_enjoyment",
    "euro12" => "depression_symptom_tearfulness",
    "ph011d10" => "psychotropic_drug_use",
    "ph006d16" => "cognitive_problems_diagnosed",

	# quality of life
    "ac012_" => "life_satisfaction",
    "mh037_" => "loneliness",

	# negative cognitive style
    "ac014_" => "age_prevents_doing_things",
    "ac015_" => "out_of_control",
    "ac016_" => "feel_left_out",
    "ac017_" => "do_things_you_want",
    "ac018_" => "family_responsibilities_prevent",
    "ac019_" => "shortage_of_money_stops",
    "ac020_" => "look_forward_each_day",
    "ac021_" => "life_has_meaning",
    "ac022_" => "look_back_with_happiness",
    "ac023_" => "feel_full_of_energy",
    "ac024_" => "full_of_opportunities",
    "ac025_" => "future_looks_good",

	# Section 3: Physical Health
    "ph071_1" => "heart_attack",
    "ph071_2" => "stroke_vascular_disease",
    "ph071_3" => "cancer",
    "ph071_4" => "hip_fracture_femoral",
    "ph072_1" => "had_condition_heart_attack",
    "ph072_2" => "had_condition_stroke_vascular_disease",
    "ph072_3" => "had_condition_cancer",
    "ph072_4" => "had_condition_hip_fracture_femoral",
    "ph073_1" => "had_condition_checked_heart_attack",
    "ph073_2" => "had_condition_checked_stroke_vascular_disease",
    "ph073_3" => "had_condition_checked_cancer",
    "ph073_4" => "had_condition_checked_hip_fracture_femoral",
    "ph074_1" => "reason_heart_attack",
    "ph074_2" => "reason_stroke_vascular_disease",
    "ph074_3" => "reason_cancer",
    "ph074_4" => "reason_hip_fracture_femoral",
    "ph075_1" => "had_condition_conf_heart_attack",
    "ph075_2" => "had_condition_conf_stroke_vascular_disease",
    "ph075_3" => "had_condition_conf_cancer",
    "ph075_4" => "had_condition_conf_hip_fracture_femoral",
    "ph076_1" => "most_recent_year_heart_attack",
    "ph076_2" => "most_recent_year_stroke_vascular_disease",
    "ph076_3" => "most_recent_year_cancer",
    "ph076_4" => "most_recent_year_hip_fracture_femoral",
    "ph077_1" => "month_condition_heart_attack",
    "ph077_2" => "month_condition_stroke_vascular_disease",
    "ph077_3" => "month_condition_cancer",
    "ph077_4" => "month_condition_hip_fracture_femoral",
    "ph006d5" => "diabetes",
    "ph006d6" => "chronic_lung_disease",
    "ph006d12" => "parkinson_disease",

	# drug use
    "ph011d1" => "drugs_high_cholesterol",
    "ph011d11" => "drugs_osteoporosis",
    "ph011d13" => "drugs_stomach_burns",
    "ph011d14" => "drugs_chronic_bronchitis",
    "ph011d15" => "drugs_corticosteroids",
    "ph011d2" => "drugs_high_blood_pressure",
    "ph011d3" => "drugs_coronary_diseases",
    "ph011d4" => "drugs_other_heart_diseases",
    "ph011d6" => "drugs_diabetes",
    "ph011d7" => "drugs_joint_pain",
    "ph011d8" => "drugs_other_pain",
    "ph011d9" => "drugs_sleep_problems",
    "ph011dno" => "drugs_none",
    "ph011dot" => "drugs_other",
    "ph003_" => "perceived_health",
    "ph006d3" => "hypercholesterolemia",

	# physical condition
    "ph046_" => "hearing",
    "ph043_" => "eyesight_distance",
    "ph044_" => "eyesight_reading",
    "ph092_" => "missing_teeth",
    "bmi2" => "bmi_categories",
    "ph012_" => "weight",
    "ph013_" => "height",
    "ph065_" => "weight_loss",
    "ph061_" => "lim_paid_work",
    "ph066_" => "reason_lost_weight",

	# ADLs (Activities of Daily Living)
    "ph041_" => "use_glasses",
    "ph045_" => "use_hearing_aid",
    "ph048d1" => "difficulty_walking_100m",
    "ph048d2" => "difficulty_sitting_2h",
    "ph048d3" => "difficulty_getting_up_chair",
    "ph048d4" => "difficulty_climbing_several_stairs",
    "ph048d5" => "difficulty_climbing_one_flight",
    "ph048d6" => "difficulty_stooping_kneeling",
    "ph048d7" => "difficulty_reaching_arms",
    "ph048d8" => "difficulty_pushing_large_objects",
    "ph048d9" => "difficulty_lifting_5kg",
    "ph048d10" => "difficulty_picking_coin",
    "ph048dno" => "difficulty_none_adl",
    "ph090_" => "bifoc_glass_lenses",
    "ph091_" => "all_natural_teeth",
    "ph094_" => "artificial_teeth",
    "ph095_" => "lost_weight",

	# IADLs (Instrumental Activities of Daily Living)
    "ph049d1" => "difficulty_dressing",
    "ph049d2" => "difficulty_walking_room",
    "ph049d3" => "difficulty_bathing",
    "ph049d4" => "difficulty_eating",
    "ph049d5" => "difficulty_getting_out_bed",
    "ph049d6" => "difficulty_using_toilet",
    "ph049d7" => "difficulty_using_map",
    "ph049d8" => "difficulty_preparing_meal",
    "ph049d9" => "difficulty_shopping",
    "ph049d10" => "difficulty_telephone",
    "ph049d11" => "difficulty_taking_medications",
    "ph049d12" => "difficulty_housework",
    "ph049d13" => "difficulty_managing_money",
    "ph049dno" => "difficulty_none_iadl",

	# frailities
    "ph089d1" => "bothered_falling_down",
    "ph089d2" => "fear_of_falling",
    "ph089d3" => "dizziness_faints_blackouts",

    # physical symptoms
    "ph084_" => "pain",
    "ph089d4" => "fatigue_frailty",
    "ph085_" => "pain_level",
    "ph008d11" => "trouble_sleeping",
    "ph008d12" => "trouble_falling_asleep",
    "ph008d13" => "trouble_waking_during_night",
    "ph008d14" => "trouble_waking_too_early",
    "ph008d15" => "trouble_feeling_restored",
    "ph008d16" => "trouble_sleeping_difficulty",
    "ph008d17" => "trouble_sleeping_tired",
    "ph008d18" => "trouble_sleeping_energy",
    "ph008d19" => "trouble_sleeping_problem",
    "ph008d20" => "trouble_sleeping_restless",
    "ph008d21" => "trouble_sleeping_insomnia",
    "ph008d22" => "trouble_sleeping_other",
    "ph008dot" => "trouble_sleeping_none",
    "ph006d1" => "heart_disease",
    "ph006d2" => "hypertension",
    "ph006d4" => "vascular_disease",
    "ph006d10" => "told_cancer",
    "ph006d11" => "ulcer",
    "ph006d13" => "cataracts",
    "ph006d14" => "femoral_fracture",
    "ph006d15" => "other_fracture",
    "ph006d16" => "alzheimer",
    "ph006d18" => "emotional_disorders",
    "ph006d19" => "rheumatoid_arthritis",
    "ph006d20" => "osteoarthritis",
    "ph006dno" => "no_disease",
    "ph006dot" => "other_disease",
    "ph004_" => "long_term_illness_disability",
    "ph005_" => "limited_activity",
    "ph008d2"  => "oral_cancer",
    "ph008d3"  => "larynx_cancer",
    "ph008d4"  => "pharynx_cancer",
    "ph008d5"  => "thryoid_cancer",
    "ph008d6"  => "lung_cancer",
    "ph008d7"  => "breast_cancer",
    "ph008d8"  => "oesophagus_cancer",
    "ph008d9"  => "stomach_cancer",
    "ph008d10" => "liver_cancer",
    "co007_" => "household_ends_meet",

	# habits and lifestyle
    "br015_" => "vigorous_physical_activity",
    "br016_" => "moderate_physical_activity",
    "phactiv" => "no_physical_activity",
)


# ╔═╡ 276b5cc2-b96c-4ddb-83dc-4ffb71978982
md"""
!!! warning
	The collection below, `dropped_variables`, is a collection of attributes that are not exploited in the work mentioned above. We decide to ignore them.
"""

# ╔═╡ a0e4c864-da20-4020-8f90-5f311ec1ba00
dropped_variables = [ "mergeid", "hhid5", "hhid6", "hhid7", "mergeidp5", "mergeidp6", "mergeidp7", "coupleid5", "coupleid6", "coupleid7", "wave", "ph008d1", "ph054_", "ph080d1", "ph080d2", "ph080d3", "ph080d4", "ph080d5", "ph080d6", "ph080d7", "ph080d8", "ph080d9", "ph080d10", "ph080d11", "ph080d12", "ph080d13", "ph080d14", "ph080d15", "ph080d16", "ph080d17", "ph080d18", "ph080d19", "ph080d20", "ph080d21", "ph080d22", "ph080dot", "ph087d1", "ph087d2", "ph087d3", "ph087d4", "ph087d5", "ph087d6", "ph087d7", "ph088_", "ph089dno", "ph082_", "ph006d21", "ph049d14", "ph049d15", "ph050_", "ph051_", "ph059d1", "ph059d2", "ph059d3", "ph059d4", "ph059d5", "ph059d6", "ph059d7", "ph059d8", "ph059d9", "ph059d10", "ph059dno", "ph059dot", "ph690d1", "ph690d2", "ph690d3", "ph690d4", "ph745_", "ph009_1", "ph009_2", "ph009_3", "ph009_4", "ph009_5", "ph009_6", "ph009_10", "ph009_11", "ph009_12", "ph009_13", "ph009_14", "ph009_15", "ph009_16", "ph009_18", "ph009_19", "ph009_20", "ph009_other", "hc012_", "hc029_", "hc114_", "hc115_", "hc125_", "ph009_21"]

# ╔═╡ e550be57-eab2-4064-b3fd-2b87c45cecbd
df = select(df_raw, Not(dropped_variables))

# ╔═╡ 6a09514b-b065-448e-bf78-420c0d6ce6a5
CSV.write(CSV_PATH, df)

# ╔═╡ 35c37143-19be-4504-a289-6404c794c617
rename!(df, Dict(Symbol(k) => Symbol(v) for (k, v) in attribute_names))

# ╔═╡ e6228654-1fd3-433d-8cc3-0ebda46e90af
n_rows, n_cols = size(df)

# ╔═╡ b7dc3954-e7e2-4064-b96b-0f18ac20fd45
@bind colname Select(names(df))

# ╔═╡ 44130ea6-884e-45a7-a580-290b09609a49
histogram(
	collect(df[:, colname]), 
	xlabel=colname, 
	ylabel="Count", 
	title=colname, 
	legend=false
)

# ╔═╡ 43648c1f-7874-4c65-b4e5-8dce3bb8b4c8
md"""
# Data Cleaning: Missing Values

We proceed to get rid of attributes and instances having too many missing values. 

Using two sliders, one for each case, we are going to set a threshold for how many missing value to keep at most.
"""

# ╔═╡ 8d047693-fc98-44da-8ba2-53b43c0ffb88
md"""
##### Columns Filtering

$(@bind perc_missing_col Slider(0:0.05:1, show_value=true, default=0.4))
"""


# ╔═╡ be2a50c2-21c3-48a5-8246-28fd34cad8ed
max_missing_instances = round(Int, perc_missing_col * n_rows)

# ╔═╡ a271df5b-07a9-4580-93f6-e4ae2cac527c
df_no_missing_columns = filter_df(df, :missing_cols; max_missing=max_missing_instances)

# ╔═╡ ce62c952-23d0-4a93-b96a-db65858a54e5
n_rows_df_no_missing_columns, n_cols_df_no_missing_columns = size(df_no_missing_columns)

# ╔═╡ a4d45954-765c-42c2-913c-60bbd5a9e8f4
md"""
!!! success "Columns filtering report"
	You filtered out the columns with more than the **$(round(perc_missing_col*100, digits=2))%** of missing values (that is, **$(convert(Int,max_missing_instances))** values).
	
	Now **$(n_cols_df_no_missing_columns)** columns remains. 
"""

# ╔═╡ 9889c086-f2d7-4878-b582-519af397271b
md"""
##### Rows Filtering

$(@bind perc_missing_row Slider(0:0.01:1, show_value=true, default=0.4))
"""


# ╔═╡ f9fa4262-616a-446e-8f36-1e1f4e057b6d
max_missing_along_row = round(Int, perc_missing_row * n_cols_df_no_missing_columns);

# ╔═╡ 35e96199-c896-46a7-bc63-0853ba7bbd14
df_no_missing = filter_df(df_no_missing_columns, :missing_rows; max_missing=max_missing_along_row)

# ╔═╡ c85c719a-3c99-4c49-b056-ce59535a457f
n_rows_df_no_missing, n_cols_df_no_missing = size(df_no_missing)

# ╔═╡ 77ee75b1-5a7c-4497-91f9-433aeb7d0a2a
md"""
!!! success "Rows filtering report"
	You filtered out the rows containing more than the $(round(perc_missing_row*100, digits=2))% of missing values (that is, $(convert(Int,max_missing_along_row)) columns).

	Now, $(n_rows_df_no_missing) instances remains.
"""

# ╔═╡ 3094d269-cd3d-46b6-9f32-cb99e19f3cc9
@bind df_no_missing_colname Select(names(df_no_missing))

# ╔═╡ d09c8f96-74fa-4e98-87b3-080f8d0aae02
histogram(
	collect(df_no_missing[:, df_no_missing_colname]), 
	xlabel=df_no_missing_colname, 
	ylabel="Count", 
	title=df_no_missing_colname, 
	legend=false
)

# ╔═╡ 147b3de2-5a64-4bfc-adc8-af8c9b5d25b8
md"""
# Labels Exploration

The attributes `initial_euro_d` and `euro_d` states whether a patient is depressed or non-depressed, respectively at baseline and follow-up.

It is important to note that the `euro_d` attribute (or the pair of the attributes above) is particularly important, since our final goal is to train machine learning predictors for this label (or, possibly, the pair).

A graphical inspection of the relation between the label and other attributes can be insightful.
"""

# ╔═╡ 4bf08b0e-ab51-4799-92bc-2c0991b561d3
md"""
!!! note
	Actually, `euro_d` can be considered a *feature* derived by summing together all the integer values in the `euro_d_attributes` columns (see below); if the total is greater than or equal to 4, then the patient is depressed at follow-up.
"""

# ╔═╡ 3575f98f-37b4-4504-aec0-a6e546a7fb7e
euro_d_attributes = [
	"depression_symptom_depression", 
	"depression_symptom_pessimism", 
	"depression_symptom_suicidality", 
	"depression_symptom_guilt", 
	"depression_symptom_sleep", 
	"depression_symptom_interest", 
	"depression_symptom_irritability", 
	"depression_symptom_appetite", 
	"depression_symptom_fatigue", 
	"depression_symptom_concentration", 
	"depression_symptom_enjoyment", 
	"depression_symptom_tearfulness",
]

# ╔═╡ 23fb94fc-1909-43a0-9151-c9b0874ea87c
md"""
!!! warning
	When dealing with `euro_d_X` attributes, we decide to ignore the (few) instances presenting some fields with -1 or -2 values (respectively, *don't know* and *I prefer not to answer*).

"""

# ╔═╡ 9c17d796-1b74-4fcf-99cb-02975bbd9a83
df_euro = filter_df(df_no_missing, 
	:property_rows;
	max_occurrences = 1,
	property        = x -> x < 0,
	colnames        = euro_d_attributes,
)

# ╔═╡ 590c1a09-4aac-46cc-9236-258593d16103
md"""
To really explore the informative content of `euro_d`, we can granularly consider the 12 contributions composing it (`euro_score_total`) and relate the resulting value with a certain `target` value.
"""

# ╔═╡ f087682f-2c34-4c5c-b098-936fd282371d
@bind target_attribute Select(names(df_euro))

# ╔═╡ e452edcd-8ac8-4bb7-a96a-26b2ce8d6058
euro_score_total = [
	sum([r[attribute] for attribute in euro_d_attributes])
	for r in eachrow(df_euro)
]

# ╔═╡ 98bd33fc-526e-45c7-bbb6-ad67abe05838
scatter(
	euro_score_total .+ 0.12 .* randn(length(euro_score_total)),
	df_euro[:, target_attribute];
	group=df_euro[:, "euro_d"],
	markerstrokewidth=0,
	markersize=1,
	xlabel="Euro-d depression at follow up",
	ylabel=target_attribute,
)

# ╔═╡ 660c38a6-2ad0-4e37-a083-d08b3c1d6413
md"""
Finally, we try to visualize the ratio between depressed and non-depressed patients with histograms and box plots.
"""

# ╔═╡ fe25627e-3fbf-462c-a0aa-82d0179cd9bb
@bind df_euro_colname Select(filter(n -> n != "euro_d", names(df_euro)))

# ╔═╡ 6243c7c3-8d9f-40a7-9276-bf80c92bd242
begin
	df_euro_temp = dropmissing(df_euro, [df_euro_colname, "euro_d"])
	col_data = collect(df_euro_temp[:, df_euro_colname])

	p1 = histogram(col_data; group=df_euro_temp[:, "euro_d"], xlabel=df_euro_colname, title="Histogram")
	p2 = boxplot(col_data; ylabel=df_euro_colname, title="Boxplot", legend=false)
	plot(p1, p2; layout=(1, 2))
end

# ╔═╡ d7a4f151-e158-43ec-948d-3f0f98fe0729
md"""
At this point, we can decide whether to keep the `euro_d_attributes` or remove them and only keep the aggregated information (the label to predict).

Probably, the best idea here is to just discard them. 
"""

# ╔═╡ f5fb9a1d-de7c-4a60-a3b3-6200386d44f5
df_euro_clean = select(df_euro, Not(euro_d_attributes))

# ╔═╡ 73b63a56-8d36-4995-8157-1838ab882aaa
size(df_euro_clean)

# ╔═╡ 623a6bdf-7e23-4511-a974-a38515f8716a
md"""
# Imputing Missings

As we will see when training machine learning models, they generally cannot handle missing values, and we have to *impute* them.

First of all, we need to separate categorical and numerical attributes.

Then, in the former case, we could treat the string "Missing" as a special category or leverage mode statistics.

In the second case we need numbers and we can exploit median and average statistics, other than mode. 
"""

# ╔═╡ 3a34594c-bfd1-4791-8cfb-4649b5d7f09e
md"""
!!! warning "Notice the change"
	Try to move your cursor under the attribute names, in the table below.
	Notice how **Float64?** is now sometimes updated as **CategoricalValue**.
"""

# ╔═╡ 26537045-ee8b-496f-82dc-628221894934
df_typed = filter_df(df_euro_clean, :cast;
	cast_threshold=10,
	ignore_cols=["euro_d"])

# ╔═╡ 8d913135-5ee6-407a-aedb-c55b68b3b70d
categorical_attribute_names = []

# ╔═╡ 4ec2c869-8b09-4b25-9bbe-101d632c096f
numerical_attribute_names = []

# ╔═╡ 340c97fe-837f-4563-a70e-1f04f9d02818
for name in names(df_typed)
	name == "euro_d" && continue
	if df_typed[1,name] isa CategoricalValue
		push!(categorical_attribute_names, name)
		println("Categorical: $(name)")
	else
		push!(numerical_attribute_names, name)
		println("Numeric: $(name)")
	end
end

# ╔═╡ fb5b262b-c8f3-44ed-8c3d-96fb20ad227a
@bind categorical_impute_strategy Select([mode, median])

# ╔═╡ 14fc7835-c63f-406e-984f-d5279640bf8e
for col in categorical_attribute_names
	val = categorical_impute_strategy(df_typed[!, col])
	df_typed[!, col] = coalesce.(df_typed[!, col], val)
end

# ╔═╡ 9ea155da-c7a1-46be-afdb-7bbf1bee5020
@bind numerical_impute_strategy Select([mode, mean, median])

# ╔═╡ 5bce1728-42b6-4d11-8e67-bdcbb261f758
for col in numerical_attribute_names
	val = numerical_impute_strategy(df_typed[!, col])
	df_typed[!, col] = coalesce.(df_typed[!, col], val)
end

# ╔═╡ 853494bf-dab0-4c17-8b00-62960b98cd28
md"""
# Further filterings

We proceed to remove the columns having very skewed categorical distributions, with at least a `frequency_threshold` percentage of identical values.
"""

# ╔═╡ 9bb12323-c3d8-4a40-a076-cf3c35fab32d
@bind frequency_threshold Slider(0:0.05:1, show_value=true, default=0.6)

# ╔═╡ 9bd07589-0e70-401a-aabd-11772b32aa34
df_freq_filter = filter_df(df_typed, :frequency;
	frequency_threshold=frequency_threshold,
	ignore_cols=["euro_d"])

# ╔═╡ a6b4bfef-7486-493d-b4e4-e6717d34c942
size(df_freq_filter)

# ╔═╡ cfc63ecc-56e9-4347-b9af-122b83a2f9a3
md"""
As a more refined strategy, let us compute the *entropy* ``H(X)`` along each column ``X = \{x_1, x_2, \ldots, x_n\}``.

```math 
H(X) = - \sum_{i=1}^{n} p(x_i)\,\log_2 p(x_i)
```

Entropy, in general, is a way to describe how much chaotic a system is from a physical point of view.

In many disciplines, such as computer science, electronics, statistics and data science, we interpret entropy as a measure of how much informative a communication channel, a signal, or a distribution is.

The concept is abstract at first, but try to think about this:
what does it mean for a communication channel to be completely uninformative?
"""

# ╔═╡ 85a27e1d-4635-46c7-bd14-89a6e6f088f8
md"""
!!! warning "Exercise"
	We want to compute the entropy of the following distribution ``X = \{a,a,a,b,b,b,b,b,c,c\}``

	``H(X) = \frac{3}{10}log_2(\frac{3}{10}) + \frac{5}{10}log_2(\frac{5}{10}) + \frac{2}{10}log_2(\frac{2}{10})=``
	``\quad = -0.52 - 0.5 - 0.46= -1.48``
"""

# ╔═╡ 8019cb0e-f560-4c80-a0cd-13061fe08d84
begin
    entropies = Dict(name => entropy(df_freq_filter[:, name]) for name in names(df_freq_filter))
    sorted_entropies = sort(collect(entropies), by=x -> x[2])
end

# ╔═╡ 2ea7a28e-3a74-4011-98f4-1c6a8cc639dd
@bind top_k_print Slider(1:1:25, show_value=true, default=10)

# ╔═╡ 28e85f18-0f54-459d-989c-6a971f25b15f
begin
	println("Top informative columns by entropy:")
    for (col, ent) in reverse(sorted_entropies)[1:top_k_print]
        println("$col → $ent")
    end
end

# ╔═╡ 0207e0c5-9a8a-4daf-942c-edc54aac352f
begin
	println("Top uninformative columns by entropy:")
    for (col, ent) in sorted_entropies[1:top_k_print]
        println("$col → $ent")
    end
end

# ╔═╡ 92a4c81a-8bd4-457a-86da-96be85c3fb89
@bind entropy_threshold Slider(0.2:0.01:3, show_value=true, default=1.0)

# ╔═╡ b0b97508-39a1-4f91-9975-188bbae4cb1b
begin
	p = plot(
		last.(sorted_entropies),
		xlabel="each i-th column (sorted by entropy)",
		ylabel="entropy",
		title="Entropy elbow plot",
		label="entropy"
	)
	
	hline!(
		p, 
		[entropy_threshold], 
		color=:red, 
		linewidth=2, 
		# linestyle=:dot, 
		label="cutoff"
	)
end

# ╔═╡ 3dc492e3-533d-46c0-bb4f-65496e87961d
df_ent_filter = filter_df(df_freq_filter, :entropy;
	entropy_threshold=entropy_threshold,
	ignore_cols=["euro_d"])

# ╔═╡ da8e28da-c864-4eeb-b163-e3349be49567
size(df_ent_filter)

# ╔═╡ 57a07a80-0aea-44c2-9d53-2f2f9d8c201b
md"""
We proceed to filter out specific instances being outliers for certain attributes.

For didactic purposes, let us fix the *age* attribute. Given the age of a specific instance, `x`, we are going to compute its z-score: 

```math 
z = \frac{x - \mu}{\sigma}
```

The score tells us how many standard deviations a value is from the mean age.
"""

# ╔═╡ 9d51ce42-752c-46f2-87bc-c064b952e770
LocalResource("../images/standard_deviation_diagram.png")

# ╔═╡ 94eadd14-2445-4ecc-a678-e8fc981674d8
# begin
# 	age_column = df_ent_filter[:, name_of_numeric_attribute]
# 	mu = mean(age_column)
# 	sigma = std(age_column)
# 	z = (age_column .- mu) ./ sigma
# end

begin
	age_column = df_ent_filter[:, "age"]
	z = zscore(age_column)
end

# ╔═╡ d88e83ce-a1d5-4b6d-9f49-061a307dbee4
@bind z_score_threshold Slider(0.01:0.01:4, show_value=true, default=2.0)

# ╔═╡ d5851387-9ceb-41ca-9e7a-e4064080e8cb
df_no_outliers = filter_df(df_ent_filter, :zscore;
	z_threshold=z_score_threshold,
	ignore_cols=["euro_d"]);

# ╔═╡ cca099f8-16f7-4960-bda0-ac86057be55b
plot(
	histogram(df_ent_filter[:, "age"]; title="age distribution", legend=false),
	histogram(df_no_outliers[:, "age"]; title="age within $(z_score_threshold)σ", legend=false);
	layout=(1, 2)
)

# ╔═╡ 9eee2f67-92b8-48c2-b502-73efa704562c
md"""
# Serialization

Now, we want to save the final result from which we want to learn machine learning models.

We use the verb *serialize* instead of *save*, because we decide to write a binary file which we can read from another Pluto.jl notebook, keeping a perfect snapshot of our clean data frame. 
"""

# ╔═╡ 12533c7b-ca7f-4960-b969-77ae3f0e9063
md"""
!!! tip
	We would like our "screenshot" to have an informative name, summarizing the whole data processing with an encoding.

	We should inject the following threshold values in the name of the file:

	`max_missing_instances`

	`max_missing_along_row` 

	`frequency_threshold` 
	
	`entropy_threshold` 
	
	`z_score_threhsold`
"""

# ╔═╡ f59ae421-e01c-43f1-9e15-6805f96aa746
filename = "share_clean_mc_$(max_missing_instances)_mr_$(max_missing_along_row)_ft_$(frequency_threshold)_et_$(entropy_threshold)_zs_$(z_score_threshold).jls"

# ╔═╡ b391021e-719e-412f-b13d-637fc5bcbe0c
SERIALIZE_PATH = joinpath(DATASET_FOLDER, filename)

# ╔═╡ 639e7cb7-fb43-4a26-9692-29a668f5d430
function save_files()
	# this is a little trick to create a new file
	open(SERIALIZE_PATH, "w") do f
	    println(f, "")
	end

	println("Writing to $(SERIALIZE_PATH)")
	
    serialize(SERIALIZE_PATH, df_no_outliers)
end

# ╔═╡ c709b580-e26a-42be-a5c1-bb76833efd65
@bind enable_saving Switch(; default=false)

# ╔═╡ 9cac84da-b466-47ec-bf66-1ab648755c9f
if enable_saving == true
	save_files()
end

# ╔═╡ Cell order:
# ╟─49733da1-b29a-41cd-a1dd-3d748ca70f97
# ╠═494947b5-219b-43ad-b29f-56216b3dc639
# ╟─a6c8a233-8c6f-43b2-a4fd-a0bbc6425d74
# ╟─e11b7142-8349-452e-9f8c-aaf006d90790
# ╠═e202a63d-b119-47ac-bf70-6516fb29f423
# ╠═f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
# ╟─c5e327e5-a169-4f25-a09d-d0c4f6540d9d
# ╠═6a09514b-b065-448e-bf78-420c0d6ce6a5
# ╟─d3a4de4f-10c8-4e70-9104-b47364b99179
# ╠═e550be57-eab2-4064-b3fd-2b87c45cecbd
# ╟─7db28b02-a364-4bd0-84fe-be7c2d87e2cc
# ╠═c88ab2ca-0a1a-4065-b1bb-10858e45b599
# ╟─276b5cc2-b96c-4ddb-83dc-4ffb71978982
# ╠═a0e4c864-da20-4020-8f90-5f311ec1ba00
# ╠═35c37143-19be-4504-a289-6404c794c617
# ╠═e6228654-1fd3-433d-8cc3-0ebda46e90af
# ╠═b7dc3954-e7e2-4064-b96b-0f18ac20fd45
# ╠═44130ea6-884e-45a7-a580-290b09609a49
# ╟─43648c1f-7874-4c65-b4e5-8dce3bb8b4c8
# ╟─8d047693-fc98-44da-8ba2-53b43c0ffb88
# ╠═be2a50c2-21c3-48a5-8246-28fd34cad8ed
# ╠═a271df5b-07a9-4580-93f6-e4ae2cac527c
# ╠═ce62c952-23d0-4a93-b96a-db65858a54e5
# ╟─a4d45954-765c-42c2-913c-60bbd5a9e8f4
# ╟─9889c086-f2d7-4878-b582-519af397271b
# ╠═f9fa4262-616a-446e-8f36-1e1f4e057b6d
# ╠═35e96199-c896-46a7-bc63-0853ba7bbd14
# ╠═c85c719a-3c99-4c49-b056-ce59535a457f
# ╟─77ee75b1-5a7c-4497-91f9-433aeb7d0a2a
# ╠═3094d269-cd3d-46b6-9f32-cb99e19f3cc9
# ╠═d09c8f96-74fa-4e98-87b3-080f8d0aae02
# ╟─147b3de2-5a64-4bfc-adc8-af8c9b5d25b8
# ╟─4bf08b0e-ab51-4799-92bc-2c0991b561d3
# ╟─3575f98f-37b4-4504-aec0-a6e546a7fb7e
# ╟─23fb94fc-1909-43a0-9151-c9b0874ea87c
# ╠═9c17d796-1b74-4fcf-99cb-02975bbd9a83
# ╟─590c1a09-4aac-46cc-9236-258593d16103
# ╠═f087682f-2c34-4c5c-b098-936fd282371d
# ╠═e452edcd-8ac8-4bb7-a96a-26b2ce8d6058
# ╠═98bd33fc-526e-45c7-bbb6-ad67abe05838
# ╟─660c38a6-2ad0-4e37-a083-d08b3c1d6413
# ╠═fe25627e-3fbf-462c-a0aa-82d0179cd9bb
# ╠═6243c7c3-8d9f-40a7-9276-bf80c92bd242
# ╟─d7a4f151-e158-43ec-948d-3f0f98fe0729
# ╠═f5fb9a1d-de7c-4a60-a3b3-6200386d44f5
# ╠═73b63a56-8d36-4995-8157-1838ab882aaa
# ╟─623a6bdf-7e23-4511-a974-a38515f8716a
# ╟─3a34594c-bfd1-4791-8cfb-4649b5d7f09e
# ╠═26537045-ee8b-496f-82dc-628221894934
# ╠═8d913135-5ee6-407a-aedb-c55b68b3b70d
# ╠═4ec2c869-8b09-4b25-9bbe-101d632c096f
# ╠═340c97fe-837f-4563-a70e-1f04f9d02818
# ╠═fb5b262b-c8f3-44ed-8c3d-96fb20ad227a
# ╠═14fc7835-c63f-406e-984f-d5279640bf8e
# ╠═9ea155da-c7a1-46be-afdb-7bbf1bee5020
# ╠═5bce1728-42b6-4d11-8e67-bdcbb261f758
# ╟─853494bf-dab0-4c17-8b00-62960b98cd28
# ╠═9bb12323-c3d8-4a40-a076-cf3c35fab32d
# ╠═9bd07589-0e70-401a-aabd-11772b32aa34
# ╠═a6b4bfef-7486-493d-b4e4-e6717d34c942
# ╟─cfc63ecc-56e9-4347-b9af-122b83a2f9a3
# ╟─85a27e1d-4635-46c7-bd14-89a6e6f088f8
# ╠═8019cb0e-f560-4c80-a0cd-13061fe08d84
# ╠═2ea7a28e-3a74-4011-98f4-1c6a8cc639dd
# ╠═28e85f18-0f54-459d-989c-6a971f25b15f
# ╠═0207e0c5-9a8a-4daf-942c-edc54aac352f
# ╠═92a4c81a-8bd4-457a-86da-96be85c3fb89
# ╠═b0b97508-39a1-4f91-9975-188bbae4cb1b
# ╠═3dc492e3-533d-46c0-bb4f-65496e87961d
# ╠═da8e28da-c864-4eeb-b163-e3349be49567
# ╟─57a07a80-0aea-44c2-9d53-2f2f9d8c201b
# ╠═9d51ce42-752c-46f2-87bc-c064b952e770
# ╠═94eadd14-2445-4ecc-a678-e8fc981674d8
# ╠═d88e83ce-a1d5-4b6d-9f49-061a307dbee4
# ╠═d5851387-9ceb-41ca-9e7a-e4064080e8cb
# ╠═cca099f8-16f7-4960-bda0-ac86057be55b
# ╟─9eee2f67-92b8-48c2-b502-73efa704562c
# ╟─12533c7b-ca7f-4960-b969-77ae3f0e9063
# ╠═f59ae421-e01c-43f1-9e15-6805f96aa746
# ╠═b391021e-719e-412f-b13d-637fc5bcbe0c
# ╠═639e7cb7-fb43-4a26-9692-29a668f5d430
# ╠═c709b580-e26a-42be-a5c1-bb76833efd65
# ╠═9cac84da-b466-47ec-bf66-1ab648755c9f
