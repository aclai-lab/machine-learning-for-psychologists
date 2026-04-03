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
	using PlutoUI
	using Plots
	using StatsPlots
	using CSV
	using DataFrames
	using StatFiles
	using FileIO

	using SoleFeatures
	using DataTreatments

	include(joinpath(@__DIR__, "..", "utils", "filters.jl"))
end

# ╔═╡ 49733da1-b29a-41cd-a1dd-3d748ca70f97
md"""
# Imports
"""

# ╔═╡ a6c8a233-8c6f-43b2-a4fd-a0bbc6425d74
md"""
# Data Loading
"""

# ╔═╡ e202a63d-b119-47ac-bf70-6516fb29f423
begin
	DATA_PATH = joinpath(@__DIR__, "..", "datasets", "not_onco_combined_dataset")
	SAV_PATH = "$(DATA_PATH).sav"
	CSV_PATH = "$(DATA_PATH).csv"
end

# ╔═╡ f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
df_raw = DataFrame(load(SAV_PATH))

# ╔═╡ adc7f08a-36c4-4eb8-9149-40626fca2bac
md"""
For more info [see the SHARE dataset](https://share-eric.eu/data/).
"""

# ╔═╡ 5d3d8370-1ad9-4721-99a9-d17768e8178a
begin 
	attribute_names = Dict(
	    # ── 1. SOCIODEMOGRAPHIC ──────────────────────────────────────────────────
	    "age_int" => "age",
	    "dn042_" => "gender",
	    "dn503_" => "ethnicity",
	    #"dn041_" => "education",
	    "dn014_" => "marital_status",
	    # "recent_widowhood"            
	    "iv009_" => "residence_rural_urban",
	    "hhsize" => "living_alone",
	    #"dn503" => "relocation",
	
	    # Social contacts
	    "sp002_" => "social_support_received",
	    "sp008_" => "social_support_given",
	    "ch001_" => "number_of_children",
	    "ch021_" => "number_of_grandchildren",
	    "dn034_" => "number_of_siblings",
	
	    # Employment / Economic
	    "ep005_" => "occupation_employment",
	    #"co007_" => "ends_meet",
	    #"co201_" => "afford_groceries",
	
	    # Activities
	    "ac035d1" => "voluntary_charity_work",
	    "ac035d4" => "educational_training_course",
	    "ac035d5" => "sport_social_club",
	    #"ac035d6" => "religious_organization",
	    "ac035d7" => "political_community_org",
	    "ac035d8" => "reading_books_magazines",
	    "ac035d9" => "word_number_games",
	    "ac035d10" => "cards_chess_games",
	    "ac035dno" => "no_activities",
	    "it003_" => "computer_skills",
	
	    # Removed due to missing values
	    "hh022_" => "perception_of_neighbourhood",
	    "hh025_" => "people_who_would_help",
	
	    # Removed due to not being verified
	    "hh017e" => "low_income",  
	
	    # ── 2. MENTAL HEALTH / PSYCHOLOGICAL ────────────────────────────────────
	    # Depression symptoms (EURO-D items)
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
	    #"ph009_" => "early_onset_depression", 
	    "ph011d10" => "psychotropic_drug_use",
	
	    # Cognitive
	    "ph006d16" => "cognitive_problems_diagnosed",
	
	    # Quality of life & other psychological
	    "ac012_" => "life_satisfaction",
	    "mh037_" => "loneliness",
	
	    # Aging perceptions / negative cognitive style
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
	
	    # ── 3. PHYSICAL HEALTH ──────────────────────────────────────────────────
	    # Physical illnesses
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
	
	    # Drug use (proxy for conditions)
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
	
	    # Physical condition
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
	
	    # ADLs (physical performance / disability)
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
	
	    # IADLs (functional limitations)
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
	
	    # Frailty
	    "ph089d1" => "bothered_falling_down",
	    "ph089d2" => "fear_of_falling",
	    "ph089d3" => "dizziness_faints_blackouts",
	
	    # Physical symptoms
	    "ph084_" => "pain",
	    "ph089d4" => "fatigue_frailty",
	    "ph085_" => "pain_level",
	
	    # no sure
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
	
	    # this is ok
	    "ph006d1" => "heart_disease",
	    "ph006d2" => "hypertension",
	    "ph006d3" => "hypercholesterolemia",
	    "ph006d4" => "vascular_disease",
	    "ph006d5" => "diabetes",
	    "ph006d6" => "asthma",
	    "ph006d10" => "told_cancer",
	    "ph006d11" => "ulcer",
	    "ph006d12" => "parkinson_disease",
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
	    
	
	
	    # Habits / lifestyle
	    #"br002_" => "current_smoking",
	    # "alcohol_6plus_drinks" => "br023_",
	    "br015_" => "vigorous_physical_activity",
	    "br016_" => "moderate_physical_activity",
	    "phactiv" => "no_physical_activity",
	)
	
	dropped_variables = [ "mergeid", "hhid5", "hhid6", "hhid7", "mergeidp5", "mergeidp6", "mergeidp7", "coupleid5", "coupleid6", "coupleid7", "ph008d1", "ph054_", "ph080d1", "ph080d2", "ph080d3", "ph080d4", "ph080d5", "ph080d6", "ph080d7", "ph080d8", "ph080d9", "ph080d10", "ph080d11", "ph080d12", "ph080d13", "ph080d14", "ph080d15", "ph080d16", "ph080d17", "ph080d18", "ph080d19", "ph080d20", "ph080d21", "ph080d22", "ph080dot", "ph087d1", "ph087d2", "ph087d3", "ph087d4", "ph087d5", "ph087d6", "ph087d7", "ph088_", "ph089dno", "ph082_", "ph006d21", "ph049d14", "ph049d15", "ph050_", "ph051_", "ph059d1", "ph059d2", "ph059d3", "ph059d4", "ph059d5", "ph059d6", "ph059d7", "ph059d8", "ph059d9", "ph059d10", "ph059dno", "ph059dot", "ph690d1", "ph690d2", "ph690d3", "ph690d4", "ph745_","ph009_1","ph009_2","ph009_3","ph009_4","ph009_5","ph009_6","ph009_10","ph009_11","ph009_12","ph009_13","ph009_14","ph009_15","ph009_16","ph009_18","ph009_19","ph009_20","ph009_other"]	
end

# ╔═╡ d3a4de4f-10c8-4e70-9104-b47364b99179
md"""
# Data Sanity Check
"""

# ╔═╡ e550be57-eab2-4064-b3fd-2b87c45cecbd
df = select(df_raw, Not(dropped_variables))

# ╔═╡ 6a09514b-b065-448e-bf78-420c0d6ce6a5
CSV.write(CSV_PATH, df)

# ╔═╡ 35c37143-19be-4504-a289-6404c794c617
rename!(df, Dict(Symbol(k) => Symbol(v) for (k, v) in attribute_names))

# ╔═╡ e6228654-1fd3-433d-8cc3-0ebda46e90af
n_rows, n_cols = size(df)

# ╔═╡ ebdec067-6848-46b5-a3b7-d7e8515146f8
description = summary_table(df)

# ╔═╡ b7dc3954-e7e2-4064-b96b-0f18ac20fd45
@bind colname Select(names(df))

# ╔═╡ 44130ea6-884e-45a7-a580-290b09609a49
run_task(HistogramTask(df[:,colname]))

# ╔═╡ 8d047693-fc98-44da-8ba2-53b43c0ffb88
@bind perc_missing_col Slider(0:0.05:1, show_value=true, default=1.0)

# ╔═╡ c77ca35d-d91d-440e-85f0-1926ed3848a6
max_missing_instances = round(Int, perc_missing_col * n_rows);

# ╔═╡ d3718495-98b7-47da-861e-982977b7a4cd
md"""
You are filtering the columns with more than the $(round(perc_missing_col*100, digits=2))% of missing values (that is, $(convert(Int,max_missing_instances)) values).
"""

# ╔═╡ a271df5b-07a9-4580-93f6-e4ae2cac527c
df_nmc = filter_along_dimension(df, max_missing_instances, dims=:cols);

# ╔═╡ 33a2f102-1a76-49aa-9ec2-39f981a51272
df_nmc_rows, df_nmc_cols = size(df_nmc)

# ╔═╡ 9d5b8635-8126-4a87-9111-b3970cabf595
md"""
Now $(df_nmc_cols) columns remains. 
"""

# ╔═╡ 9b0d582d-97cc-4015-a554-1933a44f2c35
@bind perc_missing_row Slider(0:0.01:1, show_value=true, default=1.0)

# ╔═╡ f9fa4262-616a-446e-8f36-1e1f4e057b6d
max_missing_along_row = round(Int, perc_missing_row * df_nmc_cols);

# ╔═╡ f101f661-dfc5-4b05-bbcf-04b72c8091bf
df_nmc[:,"initial_euro_d"]

# ╔═╡ 35e96199-c896-46a7-bc63-0853ba7bbd14
df_nmrc = filter_along_dimension(df_nmc, max_missing_along_row, dims=:rows)

# ╔═╡ 80bb34be-1be4-4fb4-97e7-4a5640bbd4de
round(232.232, digits=2)

# ╔═╡ 504e7090-f1b2-4d3e-95ee-8d4799c42bcc
md"""
You are filtering the instances containing more than the $(round(perc_missing_row*100, digits=2))% of missing values (that is, $(convert(Int,max_missing_along_row)) columns).
"""

# ╔═╡ c85c719a-3c99-4c49-b056-ce59535a457f
df_nmrc_rows, df_nmrc_cols = size(df_nmrc)

# ╔═╡ d1e0a774-4a70-4923-8206-bca327ff27d8
md"""
Now, $(df_nmrc_rows) instances remains.
"""

# ╔═╡ 3094d269-cd3d-46b6-9f32-cb99e19f3cc9
@bind df_nmrc_colname Select(names(df_nmrc))

# ╔═╡ d09c8f96-74fa-4e98-87b3-080f8d0aae02
run_task(HistogramTask(df_nmrc[:,df_nmrc_colname]))

# ╔═╡ 25bc4de2-a1df-4e78-87bd-118f33287928
y = df_nmrc[:,"euro_d"]

# ╔═╡ f84cd739-ab38-4e21-b9c1-f0447267f73e
begin
	df_nmrc[:,"initial_euro_d_int"] = [v == "no" ? 0 : 1 for v in df_nmrc[:,"initial_euro_d"]]

	select!(df_nmrc, Not("initial_euro_d"));
end

# ╔═╡ 147b3de2-5a64-4bfc-adc8-af8c9b5d25b8
md"""
# Manual check for euro scores

We decide to remove the (few) rows containing -1 and -2 in the *euro_d* fields (respectively, "Don't know" and the "I prefer not to answer").
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

# ╔═╡ 31cb1d64-9781-47c4-99a9-b1952fcfefbf
typeof.(df_nmrc[:, "depression_symptom_tearfulness"]) |> unique

# ╔═╡ 9c17d796-1b74-4fcf-99cb-02975bbd9a83
df_euro = filter_along_dimension(df_nmrc, 1; dims=:rows, property=(x -> x isa Number && (x < 0)), colnames=euro_d_attributes)

# ╔═╡ e452edcd-8ac8-4bb7-a96a-26b2ce8d6058
euro_score_total = [
	sum([r[attribute] for attribute in euro_d_attributes])
	for r in eachrow(df_euro)
]

# ╔═╡ 98bd33fc-526e-45c7-bbb6-ad67abe05838
run_task(ScatterTask(
	euro_score_total .+ 0.15 .* randn(length(euro_score_total)), 
	df_euro[:, "age"]; 
	params=(
		group=df_euro[:, "euro_d"], 
		markerstrokewidth=0,  
		jitter=0.2,
		markersize=1,
		xlabel="Euro-d depression at follow up",
		ylabel="Age",
	)
))

# ╔═╡ a246c333-3fe3-4d82-8b00-fb0cd327a956


# ╔═╡ 1e8b58a2-d470-49ed-aa23-f7eecb6f00cb
md"""
# TODO: Wrapper Filters
"""

# ╔═╡ Cell order:
# ╟─49733da1-b29a-41cd-a1dd-3d748ca70f97
# ╠═494947b5-219b-43ad-b29f-56216b3dc639
# ╟─a6c8a233-8c6f-43b2-a4fd-a0bbc6425d74
# ╠═e202a63d-b119-47ac-bf70-6516fb29f423
# ╠═f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
# ╟─adc7f08a-36c4-4eb8-9149-40626fca2bac
# ╟─5d3d8370-1ad9-4721-99a9-d17768e8178a
# ╠═6a09514b-b065-448e-bf78-420c0d6ce6a5
# ╟─d3a4de4f-10c8-4e70-9104-b47364b99179
# ╠═e550be57-eab2-4064-b3fd-2b87c45cecbd
# ╠═35c37143-19be-4504-a289-6404c794c617
# ╠═e6228654-1fd3-433d-8cc3-0ebda46e90af
# ╠═ebdec067-6848-46b5-a3b7-d7e8515146f8
# ╠═b7dc3954-e7e2-4064-b96b-0f18ac20fd45
# ╠═44130ea6-884e-45a7-a580-290b09609a49
# ╠═8d047693-fc98-44da-8ba2-53b43c0ffb88
# ╟─c77ca35d-d91d-440e-85f0-1926ed3848a6
# ╟─d3718495-98b7-47da-861e-982977b7a4cd
# ╠═a271df5b-07a9-4580-93f6-e4ae2cac527c
# ╠═33a2f102-1a76-49aa-9ec2-39f981a51272
# ╟─9d5b8635-8126-4a87-9111-b3970cabf595
# ╠═9b0d582d-97cc-4015-a554-1933a44f2c35
# ╠═f9fa4262-616a-446e-8f36-1e1f4e057b6d
# ╠═f101f661-dfc5-4b05-bbcf-04b72c8091bf
# ╠═35e96199-c896-46a7-bc63-0853ba7bbd14
# ╠═80bb34be-1be4-4fb4-97e7-4a5640bbd4de
# ╟─504e7090-f1b2-4d3e-95ee-8d4799c42bcc
# ╠═c85c719a-3c99-4c49-b056-ce59535a457f
# ╟─d1e0a774-4a70-4923-8206-bca327ff27d8
# ╠═3094d269-cd3d-46b6-9f32-cb99e19f3cc9
# ╠═d09c8f96-74fa-4e98-87b3-080f8d0aae02
# ╠═25bc4de2-a1df-4e78-87bd-118f33287928
# ╠═f84cd739-ab38-4e21-b9c1-f0447267f73e
# ╟─147b3de2-5a64-4bfc-adc8-af8c9b5d25b8
# ╠═3575f98f-37b4-4504-aec0-a6e546a7fb7e
# ╠═31cb1d64-9781-47c4-99a9-b1952fcfefbf
# ╠═9c17d796-1b74-4fcf-99cb-02975bbd9a83
# ╠═e452edcd-8ac8-4bb7-a96a-26b2ce8d6058
# ╠═98bd33fc-526e-45c7-bbb6-ad67abe05838
# ╠═a246c333-3fe3-4d82-8b00-fb0cd327a956
# ╟─1e8b58a2-d470-49ed-aa23-f7eecb6f00cb
