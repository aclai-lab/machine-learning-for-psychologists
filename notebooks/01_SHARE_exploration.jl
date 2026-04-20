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

# ╔═╡ 2807db23-4cbb-4542-9de1-26610595c6bd
md"""
# SHARE data preprocessing

Welcome! In this project, we are interested in training a machine learning model for properly estimate the risk of an individual developing depression.

In this notebook we focus in the data exploration and cleaning processes.
"""

# ╔═╡ 49733da1-b29a-41cd-a1dd-3d748ca70f97
md"""
# Imports
"""

# ╔═╡ a6c8a233-8c6f-43b2-a4fd-a0bbc6425d74
md"""
# Data Loading

In this section, we are going to load the data coming from the SHARE research infrastructure, consisting of more than 40.000 surveys.

It is important to note that the data is publicly available, but we are going to consider a slightly refined version by Murri et al., described in their work [Risk Prediction Models for Depression in Community-Dwelling Older Adults](https://www.ajgponline.org/article/S1064-7481(22)00435-3/abstract). 

In the work just mentioned, data was collected with the final goal of developing machine-learning predictors for late-life depression, including demographic characteristics, health-related factors, disability and individual depressive symptoms.

In particular, we are dealing with data from wave 5 (collected in 2013), consisting of baseline and retrospective information, and outcome data from wave 6 (collected in 2015). 
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
!!! warning "Dropped variables"
	The collection below, `dropped_variables`, is a collection of attributes that are not exploited in the work mentioned above. We decide to ignore them.
"""

# ╔═╡ a0e4c864-da20-4020-8f90-5f311ec1ba00
dropped_variables = [ "mergeid", "hhid5", "hhid6", "hhid7", "mergeidp5", "mergeidp6", "mergeidp7", "coupleid5", "coupleid6", "coupleid7", "wave", "ph008d1", "ph054_", "ph080d1", "ph080d2", "ph080d3", "ph080d4", "ph080d5", "ph080d6", "ph080d7", "ph080d8", "ph080d9", "ph080d10", "ph080d11", "ph080d12", "ph080d13", "ph080d14", "ph080d15", "ph080d16", "ph080d17", "ph080d18", "ph080d19", "ph080d20", "ph080d21", "ph080d22", "ph080dot", "ph087d1", "ph087d2", "ph087d3", "ph087d4", "ph087d5", "ph087d6", "ph087d7", "ph088_", "ph089dno", "ph082_", "ph006d21", "ph049d14", "ph049d15", "ph050_", "ph051_", "ph059d1", "ph059d2", "ph059d3", "ph059d4", "ph059d5", "ph059d6", "ph059d7", "ph059d8", "ph059d9", "ph059d10", "ph059dno", "ph059dot", "ph690d1", "ph690d2", "ph690d3", "ph690d4", "ph745_", "ph009_1", "ph009_2", "ph009_3", "ph009_4", "ph009_5", "ph009_6", "ph009_10", "ph009_11", "ph009_12", "ph009_13", "ph009_14", "ph009_15", "ph009_16", "ph009_18", "ph009_19", "ph009_20", "ph009_other", "hc012_", "hc029_", "hc114_", "hc115_", "hc125_", "ph009_21"]

# ╔═╡ 51f15ec3-d939-4416-9281-3b34b673fed6
md"""
!!! warning "Float64?"
	Most of our attributes are of type "Float64?".

	What does this mean?
"""

# ╔═╡ f5a540a6-4660-47cb-8f82-594d8d40dd1f
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

This step can be considered naive, in certain scenario (see below); we propose and discuss it anyway for educative purposes. Later, we are going to drop attributes based on more refined considerations.
"""

# ╔═╡ fee060cb-c7b4-4956-a47b-8219bd9aa3b1
md"""
!!! warning "The trap of removing missing values"
	The following steps are useful for clean the data but, depending on the domain of interest, could result in a very naive mistake!

	Given a specific attribute, the value of a certain instance can be missing for three reasons:
	- **MCAR** (Missing Completely At Random): missings are due to random events.
	- **MAR** (Missing At Random): missings depends on other observed variables but not the missing data itself (e.g., women are more likely to report their weight than men);
	- **MNAR** (Missing Not At Random): the probability of missingness depends on the missing values themselves (e.g., high-income earners refuse to disclose their salary and missingness is directly tied to salary amount).

	Dropping attributes with many MAR and MNAR can bias results badly!
"""

# ╔═╡ 8d047693-fc98-44da-8ba2-53b43c0ffb88
md"""
##### Columns Filtering

$(@bind perc_missing_col Slider(0:0.05:1, show_value=true, default=0.8))
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

$(@bind perc_missing_row Slider(0:0.01:1, show_value=true, default=0.8))
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
	You filtered out **$(n_rows_df_no_missing_columns - n_rows_df_no_missing)** rows, that is, those containing more than the **$(round(perc_missing_row*100, digits=2))%** of missing values (i.e., **$(convert(Int,max_missing_along_row)) columns** are missing).

	Now, **$(n_rows_df_no_missing)** instances remains.
"""

# ╔═╡ 5d02a4a5-7173-43ae-8422-1bf5e245529c
md"""
!!! info "Exercise"
	Try to investigate the following attributes. Which one seems to be "informative"? Which are garbage, and why? Which columns are intuitively useless?

	- `residence_rural_urban`
	- `low_income` 
	- `number_of_children`
	- `number_of_grandchildren`
	- `political_community_org`
	- `hearing`
	- `ethnicity`
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

# ╔═╡ 4ec2c869-8b09-4b25-9bbe-101d632c096f
numerical_attribute_names = []

# ╔═╡ 8d913135-5ee6-407a-aedb-c55b68b3b70d
categorical_attribute_names = []

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

# ╔═╡ 9ea155da-c7a1-46be-afdb-7bbf1bee5020
@bind numerical_impute_strategy Select([mode, mean, median])

# ╔═╡ 5bce1728-42b6-4d11-8e67-bdcbb261f758
for col in numerical_attribute_names
	val = numerical_impute_strategy(df_typed[!, col])
	df_typed[!, col] = coalesce.(df_typed[!, col], val)
end

# ╔═╡ 97f45a63-5148-439f-b4bf-10d371dc357a
md"""
!!! info "Exercise"
	*In this case*, should we just keep the missing values or is it better to impute them using the mode?
	If the latter were the case, why could we not exploit mean and median metrics?
"""

# ╔═╡ 583f660f-e955-4ef0-bf04-555b1e294b7e
function just_return_missing(_)
	return missing
end

# ╔═╡ fb5b262b-c8f3-44ed-8c3d-96fb20ad227a
@bind categorical_impute_strategy Select([just_return_missing, mode])

# ╔═╡ 14fc7835-c63f-406e-984f-d5279640bf8e
for col in categorical_attribute_names
	val = categorical_impute_strategy(df_typed[!, col])
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
	ignore_cols=["euro_d"]);

# ╔═╡ a6b4bfef-7486-493d-b4e4-e6717d34c942
size(df_freq_filter)

# ╔═╡ 62744abc-80f2-49aa-9a49-f371a4427194
md"""
!!! success "OK"
	The size went from **$(size(df_typed))** to **$(size(df_freq_filter))**.
"""

# ╔═╡ 401f93f6-9253-43ba-8957-83f214fde4f0
md"""
### Monovariate approach
"""

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
	age_column = df_freq_filter[:, "age"]
	z = zscore(age_column)
end

# ╔═╡ d88e83ce-a1d5-4b6d-9f49-061a307dbee4
@bind z_score_threshold Slider(0.01:0.01:4, show_value=true, default=2.0)

# ╔═╡ d5851387-9ceb-41ca-9e7a-e4064080e8cb
df_no_outliers = filter_df(df_freq_filter, :zscore;
	z_threshold=z_score_threshold,
	ignore_cols=["euro_d"]);

# ╔═╡ cca099f8-16f7-4960-bda0-ac86057be55b
plot(
	histogram(df_freq_filter[:, "age"]; title="age distribution", legend=false),
	histogram(df_no_outliers[:, "age"]; title="age within $(z_score_threshold)σ", legend=false);
	layout=(1, 2)
)

# ╔═╡ aca83632-41ae-4696-ba82-7bcbbc5c6571
size(df_no_outliers)

# ╔═╡ f7a5338f-9652-49f6-a95b-a91477e0788a
md"""
### Multivariate approach
"""

# ╔═╡ cfc63ecc-56e9-4347-b9af-122b83a2f9a3
md"""
As a more refined filtering strategy, let us compute the *entropy* ``H(X)`` along each column ``X = \{x_1, x_2, \ldots, x_n\}``.

```math 
H(X) = - \sum_{i=1}^{n} p(x_i)\,\log_2 p(x_i)
```

Entropy, in general, is a way to describe how much chaotic a system is from a physical point of view.

In many disciplines, such as computer science, electronics, statistics and data science, we interpret entropy as a measure of how much informative a communication channel, a signal, or a distribution is.
"""

# ╔═╡ 35aa1f50-a2db-42c5-a080-45ec0d76ee33
md"""
!!! warning "Exercise"
	We want to compute the entropy of the following distribution ``X = \{a,a,a,b,b,b,b,b,c,c\}``

	``H(X) = -(\frac{3}{10}log_2(\frac{3}{10}) + \frac{5}{10}log_2(\frac{5}{10}) + \frac{2}{10}log_2(\frac{2}{10}))=``

	``\quad = -(-0.52 - 0.5 - 0.46)= 1.48``
"""

# ╔═╡ 2380ce25-6af6-4ed3-b528-d4fd55ef4abb
md"""
!!! info "Exercise"
	Tell if this sentence is correct or not.

	*If we compute the entropy for an attribute, and it is relatively low, then this means that the attribute is not informative and can be discarded*.
"""

# ╔═╡ b4d57562-aab5-44ee-be52-3c106e0ce170
md"""
Knowing the entropy of two columns, X and Y, we can compute the *mutual information* (MI):

```math
I(X, Y) = H(Y) - H(Y | X)
```

This measures how much knowing X reduces uncertainty about Y. 
In other words, MI is a measure of the amount of information that one random variable contains about another random variable.

In our scenario, the mutual information gives as an important insight about how much an attribute is useful in predicting the class label.
"""

# ╔═╡ 85a27e1d-4635-46c7-bd14-89a6e6f088f8
md"""
!!! warning "Exercise"
	We want to compute the mutual information between ``X = \{a,a,a,b,b,b,b,b,c,c\}`` and ``Y = \{0, 0, 1, 1, 1, 1, 0, 1, 0, 0\}``.

	``H(Y) = -(\frac{4}{10}log_2(\frac{4}{10}) + \frac{6}{10}log_2(\frac{6}{10}))= 0.97``

	``H(Y|X=a) = -(\frac{2}{3}log_2(\frac{2}{3}) + \frac{1}{3}log_2(\frac{1}{3}))= 0.91``

	``H(Y|X=b) = -(\frac{1}{5}log_2(\frac{1}{5}) + \frac{4}{5}log_2(\frac{4}{5}))= 0.72``

	``H(Y|X=c) = -(\frac{2}{2}log_2(\frac{2}{2}) = 0``

	``H(Y|X) = P(X=a) * H(Y|X=a) + ... + P(X=c) * H(Y | X=c)``

	``\quad = 0.3 * 0.91 + 0.5 * 0.722 + 0 = 0.636``

	``I(X, Y) = 0.970 - 0.636 = 0.334``
"""

# ╔═╡ 240443a3-42f2-4baf-b46b-621b97d5cd51
mi_dict = Dict(
	name => mutual_information(
		df_no_outliers[:, name], 
		df_no_outliers[:, "euro_d"];
	) 
	for name in names(df_no_outliers)
	if name != "euro_d"
)

# ╔═╡ 7e3de156-3690-4ec2-b138-093f239f4a32
mi_dict_sorted = sort(
	collect(mi_dict),
	by=x -> x[2]
)

# ╔═╡ 2ea7a28e-3a74-4011-98f4-1c6a8cc639dd
@bind top_k_print Slider(1:1:25, show_value=true, default=10)

# ╔═╡ 28e85f18-0f54-459d-989c-6a971f25b15f
begin
	println("Top $(top_k_print) mutual information:")
    for (col, ent) in reverse(mi_dict_sorted)[1:top_k_print]
        println("$col → $ent")
    end
end

# ╔═╡ 0207e0c5-9a8a-4daf-942c-edc54aac352f
begin
	println("Top $(top_k_print) mutual information:")
    for (col, ent) in mi_dict_sorted[1:top_k_print]
        println("$col → $ent")
    end
end

# ╔═╡ 92a4c81a-8bd4-457a-86da-96be85c3fb89
@bind mi_threshold Slider(0.0:0.001:0.03, show_value=true, default=0.01)

# ╔═╡ b0b97508-39a1-4f91-9975-188bbae4cb1b
begin
	p = plot(
		last.(mi_dict_sorted),
		xlabel="i-th attribute with lower mutual information",
		ylabel="mutual information",
		title="Mutual information elbow plot",
		label="m.i. with class",
		ylims=(0, 0.1),
		legend=:topleft
	)
	
	hline!(
		p, 
		[mi_threshold], 
		color=:red, 
		linewidth=2, 
		# linestyle=:dot, 
		label="cutoff"
	)
end

# ╔═╡ 3dc492e3-533d-46c0-bb4f-65496e87961d
df_mi_filter = filter_df(df_no_outliers, :information;
    information_dictionary=mi_dict,
	information_threshold=mi_threshold,
	ignore_cols=["euro_d"]);

# ╔═╡ da8e28da-c864-4eeb-b163-e3349be49567
size(df_mi_filter)

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
filename = "share_clean_mc_$(max_missing_instances)_mr_$(max_missing_along_row)_ft_$(frequency_threshold)_mit_$(mi_threshold)_zs_$(z_score_threshold).jls"

# ╔═╡ b391021e-719e-412f-b13d-637fc5bcbe0c
SERIALIZE_PATH = joinpath(DATASET_FOLDER, filename)

# ╔═╡ 639e7cb7-fb43-4a26-9692-29a668f5d430
function save_files()
	# this is a little trick to create a new file
	open(SERIALIZE_PATH, "w") do f
	    println(f, "")
	end

	println("Writing to $(SERIALIZE_PATH)")
	
    serialize(SERIALIZE_PATH, df_mi_filter)
end

# ╔═╡ 5489432a-68e4-49b1-aa70-d2ee410d5dd1
md"""
!!! tips "Ready to go!"
	Press the slider below to finally save your dataset.
	We strongly suggest pressing the slider again a few seconds later, to avoid saving a new dataset every time this notebook is updated.
"""

# ╔═╡ c709b580-e26a-42be-a5c1-bb76833efd65
@bind enable_saving Switch(; default=false)

# ╔═╡ 9cac84da-b466-47ec-bf66-1ab648755c9f
if enable_saving == true
	save_files()
end

# ╔═╡ 4a5dd797-5fb3-4074-9c7e-02c8db096777
md"""
# A final exercise about leveraging LLMs

### First of all, an important definition: Epistemia
*Epistemia* is a neologism describing the illusion of having knowledge created by interacting with generative AI, where **fluent**, **coherent**, and **convincing** text makes information seem reliable even when it may **lack a real factual foundation**.

Epistemia is about people confusing the appearance of understanding produced by large language models with verified knowledge.

In other words, epistemia is when the appearance of knowledge replaces genuine epistemic reliability.
"""

# ╔═╡ 660e2581-0636-4162-9477-855dcd6030b9
md"""
!!! info "A first reflection"
	1. Before the advent of large generative models, what was the most similar phenomenon with respect to epistemia?

	2. What is the *crucial* aspect that changed, from the idea of your previous answer to nowadays?

	3. How can we defend ourselves from epistemia?
"""

# ╔═╡ 33024b81-50ac-45e4-8168-0a842c4d522d
md"""
### Now, let's challenge LLMs
In the following section, we are going to ask for clarification to ChatGPT, tackling many aspects we studied during the lesson.
"""

# ╔═╡ c3af66c9-0998-4952-9e5b-91076763a922
md"""
### Prompt #1
```
I want to analyse the SHARE European dataset.
Tell me the meaning of these variable names.

"mergeid", "age_int", "hhsize", "dn042_", "dn503_", "dn014_", "dn034_", "iv009_", "hh022_", "hh025_", "hh017e", "sp002_", "sp008_", "ch001_", "ch021_", "ep005_", "co007_", "ac035d1", "ac035d4", "ac035d5", "ac035d7", "ac035d8", "ac035d9", "ac035d10", "ac035dno", "ac012_", "ac014_", "ac015_", "ac016_", "ac017_", "ac018_", "ac019_", "ac020_", "ac021_", "ac022_", "ac023_", "ac024_", "ac025_", "it003_", "euro1", "euro2", "euro3", "euro4", "euro5", "euro6", "euro7", "euro8", "euro9", "euro10", "euro11", "euro12", "bmi2", "phactiv", "hhid5", "mergeidp5", "coupleid5", "country", "language", "ph003_", "ph004_", "ph005_", "ph006d1", "ph006d2", "ph006d3", "ph006d4", "ph006d5", "ph006d6", "ph006d10", "ph006d11", "ph006d12", "ph006d13", "ph006d14", "ph006d15", "ph006d16", "ph006d18", "ph006d19", "ph006d20", "ph006dno", "ph006dot", "ph008d1", "ph008d2", "ph008d3", "ph008d4", "ph008d5", "ph008d6", "ph008d7", "ph008d8", "ph008d9", "ph008d10", "ph008d11", "ph008d12", "ph008d13", "ph008d14", "ph008d15", "ph008d16", "ph008d17", "ph008d18", "ph008d19", "ph008d20", "ph008d21", "ph008d22", "ph008dot", "ph009_1", "ph009_2", "ph009_3", "ph009_4", "ph009_5", "ph009_6", "ph009_10", "ph009_11", "ph009_12", "ph009_13", "ph009_14", "ph009_15", "ph009_16", "ph009_18", "ph009_19", "ph009_20", "ph009_other", "ph011d1", "ph011d2", "ph011d3", "ph011d4", "ph011d6", "ph011d7", "ph011d8", "ph011d9", "ph011d10", "ph011d11", "ph011d13", "ph011d14", "ph011d15", "ph011dno", "ph011dot", "ph012_", "ph013_", "ph041_", "ph043_", "ph044_", "ph045_", "ph046_", "ph048d1", "ph048d2", "ph048d3", "ph048d4", "ph048d5", "ph048d6", "ph048d7", "ph048d8", "ph048d9", "ph048d10", "ph048dno", "ph049d1", "ph049d2", "ph049d3", "ph049d4", "ph049d5", "ph049d6", "ph049d7", "ph049d8", "ph049d9", "ph049d10", "ph049d11", "ph049d12", "ph049d13", "ph049dno", "ph054_", "ph061_", "ph065_", "ph066_", "ph071_1", "ph071_2", "ph071_3", "ph071_4", "ph072_1", "ph072_2", "ph072_3", "ph072_4", "ph073_1", "ph073_2", "ph073_3", "ph073_4", "ph074_1", "ph074_2", "ph074_3", "ph074_4", "ph075_1", "ph075_2", "ph075_3", "ph075_4", "ph076_1", "ph076_2", "ph076_3", "ph076_4", "ph077_1", "ph077_2", "ph077_3", "ph077_4", "ph080d1", "ph080d2", "ph080d3", "ph080d4", "ph080d5", "ph080d6", "ph080d7", "ph080d8", "ph080d9", "ph080d10", "ph080d11", "ph080d12", "ph080d13", "ph080d14", "ph080d15", "ph080d16", "ph080d17", "ph080d18", "ph080d19", "ph080d20", "ph080d21", "ph080d22", "ph080dot", "ph084_", "ph085_", "ph087d1", "ph087d2", "ph087d3", "ph087d4", "ph087d5", "ph087d6", "ph087d7", "ph088_", "ph089d1", "ph089d2", "ph089d3", "ph089d4", "ph089dno", "ph090_", "ph091_", "ph092_", "ph094_", "ph095_", "mh037_", "hc012_", "hc029_", "hc114_", "hc115_", "hc125_", "br015_", "br016_", "isced1997_r", "wave", "initial_euro_d", "euro_d", "hhid6", "mergeidp6", "coupleid6", "ph006d21", "ph009_21", "ph049d14", "ph049d15", "ph050_", "ph051_", "ph059d1", "ph059d2", "ph059d3", "ph059d4", "ph059d5", "ph059d6", "ph059d7", "ph059d8", "ph059d9", "ph059d10", "ph059dno", "ph059dot", "ph082_", "ph690d1", "ph690d2", "ph690d3", "ph690d4", "hhid7", "mergeidp7", "coupleid7", "ph745_"
```
"""

# ╔═╡ 8833e136-07dd-42ff-b5cc-cb4cff87c194
LocalResource("../images/prompt-01-01.png")

# ╔═╡ 1741894c-87e8-40c0-bc33-5086e5823a59
md"""
Actually, in both for [Wave 5](https://www.share-datadocutool.org/control-construct-schemes/view/99) and [Wave 6](https://www.share-datadocutool.org/control-construct-schemes/view/145), `hh022_` is the question *I really feel part of this area. Would you say you strongly agree, agree, disagree or strongly disagree?* 
"""

# ╔═╡ 725e38d1-8f98-4a34-8ae8-fc134b2e490f
md"""
---
"""

# ╔═╡ e1ad8296-dd83-4ce7-9e25-779e81d06727
md"""
### Prompt #2
I am writing a data analysis pipeline. I have a lot of missing values and I want to remove them! Show me how to do it in Julia
"""

# ╔═╡ 3ed03a33-1a2d-4f38-ad69-e3db2f5294bb
LocalResource("../images/prompt-01-02.png")

# ╔═╡ 2f0c0e80-7ed1-444a-9387-bb289901acf6
md"""
---
"""

# ╔═╡ 2a5776a9-0732-4c0b-8227-8e42e664b7f4
md"""
### Prompt #3
I am doing a data analysis in Julia and I want to know more about monovariate filter. Recently I heard about a method based on a metric called "entropy", I want to use that.
"""

# ╔═╡ 44d4af15-b64b-4b8c-8e7f-f72b33f3c53b
LocalResource("../images/prompt-01-03.png")

# ╔═╡ Cell order:
# ╟─2807db23-4cbb-4542-9de1-26610595c6bd
# ╟─49733da1-b29a-41cd-a1dd-3d748ca70f97
# ╠═494947b5-219b-43ad-b29f-56216b3dc639
# ╟─a6c8a233-8c6f-43b2-a4fd-a0bbc6425d74
# ╟─e11b7142-8349-452e-9f8c-aaf006d90790
# ╠═e202a63d-b119-47ac-bf70-6516fb29f423
# ╠═f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
# ╟─c5e327e5-a169-4f25-a09d-d0c4f6540d9d
# ╠═6a09514b-b065-448e-bf78-420c0d6ce6a5
# ╟─d3a4de4f-10c8-4e70-9104-b47364b99179
# ╟─7db28b02-a364-4bd0-84fe-be7c2d87e2cc
# ╟─c88ab2ca-0a1a-4065-b1bb-10858e45b599
# ╟─276b5cc2-b96c-4ddb-83dc-4ffb71978982
# ╟─a0e4c864-da20-4020-8f90-5f311ec1ba00
# ╟─51f15ec3-d939-4416-9281-3b34b673fed6
# ╟─f5a540a6-4660-47cb-8f82-594d8d40dd1f
# ╠═35c37143-19be-4504-a289-6404c794c617
# ╠═e6228654-1fd3-433d-8cc3-0ebda46e90af
# ╠═b7dc3954-e7e2-4064-b96b-0f18ac20fd45
# ╠═44130ea6-884e-45a7-a580-290b09609a49
# ╟─43648c1f-7874-4c65-b4e5-8dce3bb8b4c8
# ╟─fee060cb-c7b4-4956-a47b-8219bd9aa3b1
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
# ╟─5d02a4a5-7173-43ae-8422-1bf5e245529c
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
# ╠═4ec2c869-8b09-4b25-9bbe-101d632c096f
# ╠═8d913135-5ee6-407a-aedb-c55b68b3b70d
# ╠═340c97fe-837f-4563-a70e-1f04f9d02818
# ╠═9ea155da-c7a1-46be-afdb-7bbf1bee5020
# ╠═5bce1728-42b6-4d11-8e67-bdcbb261f758
# ╟─97f45a63-5148-439f-b4bf-10d371dc357a
# ╠═583f660f-e955-4ef0-bf04-555b1e294b7e
# ╠═fb5b262b-c8f3-44ed-8c3d-96fb20ad227a
# ╠═14fc7835-c63f-406e-984f-d5279640bf8e
# ╟─853494bf-dab0-4c17-8b00-62960b98cd28
# ╠═9bb12323-c3d8-4a40-a076-cf3c35fab32d
# ╠═9bd07589-0e70-401a-aabd-11772b32aa34
# ╠═a6b4bfef-7486-493d-b4e4-e6717d34c942
# ╟─62744abc-80f2-49aa-9a49-f371a4427194
# ╠═401f93f6-9253-43ba-8957-83f214fde4f0
# ╟─57a07a80-0aea-44c2-9d53-2f2f9d8c201b
# ╠═9d51ce42-752c-46f2-87bc-c064b952e770
# ╠═94eadd14-2445-4ecc-a678-e8fc981674d8
# ╠═d88e83ce-a1d5-4b6d-9f49-061a307dbee4
# ╠═d5851387-9ceb-41ca-9e7a-e4064080e8cb
# ╠═cca099f8-16f7-4960-bda0-ac86057be55b
# ╠═aca83632-41ae-4696-ba82-7bcbbc5c6571
# ╠═f7a5338f-9652-49f6-a95b-a91477e0788a
# ╟─cfc63ecc-56e9-4347-b9af-122b83a2f9a3
# ╟─35aa1f50-a2db-42c5-a080-45ec0d76ee33
# ╟─2380ce25-6af6-4ed3-b528-d4fd55ef4abb
# ╟─b4d57562-aab5-44ee-be52-3c106e0ce170
# ╟─85a27e1d-4635-46c7-bd14-89a6e6f088f8
# ╠═240443a3-42f2-4baf-b46b-621b97d5cd51
# ╠═7e3de156-3690-4ec2-b138-093f239f4a32
# ╠═2ea7a28e-3a74-4011-98f4-1c6a8cc639dd
# ╟─28e85f18-0f54-459d-989c-6a971f25b15f
# ╠═0207e0c5-9a8a-4daf-942c-edc54aac352f
# ╠═92a4c81a-8bd4-457a-86da-96be85c3fb89
# ╠═b0b97508-39a1-4f91-9975-188bbae4cb1b
# ╠═3dc492e3-533d-46c0-bb4f-65496e87961d
# ╠═da8e28da-c864-4eeb-b163-e3349be49567
# ╟─9eee2f67-92b8-48c2-b502-73efa704562c
# ╟─12533c7b-ca7f-4960-b969-77ae3f0e9063
# ╠═f59ae421-e01c-43f1-9e15-6805f96aa746
# ╠═b391021e-719e-412f-b13d-637fc5bcbe0c
# ╠═639e7cb7-fb43-4a26-9692-29a668f5d430
# ╟─5489432a-68e4-49b1-aa70-d2ee410d5dd1
# ╠═c709b580-e26a-42be-a5c1-bb76833efd65
# ╠═9cac84da-b466-47ec-bf66-1ab648755c9f
# ╟─4a5dd797-5fb3-4074-9c7e-02c8db096777
# ╟─660e2581-0636-4162-9477-855dcd6030b9
# ╟─33024b81-50ac-45e4-8168-0a842c4d522d
# ╟─c3af66c9-0998-4952-9e5b-91076763a922
# ╠═8833e136-07dd-42ff-b5cc-cb4cff87c194
# ╟─1741894c-87e8-40c0-bc33-5086e5823a59
# ╟─725e38d1-8f98-4a34-8ae8-fc134b2e490f
# ╟─e1ad8296-dd83-4ce7-9e25-779e81d06727
# ╟─3ed03a33-1a2d-4f38-ad69-e3db2f5294bb
# ╟─2f0c0e80-7ed1-444a-9387-bb289901acf6
# ╟─2a5776a9-0732-4c0b-8227-8e42e664b7f4
# ╟─44d4af15-b64b-4b8c-8e7f-f72b33f3c53b
