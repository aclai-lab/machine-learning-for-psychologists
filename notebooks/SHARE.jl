### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 494947b5-219b-43ad-b29f-56216b3dc639
begin
	import Pkg
	Pkg.activate(Base.current_project(@__DIR__))
	using PlutoUI
	using CSV
	using DataFrames
	using StatFiles
	using FileIO

	using SoleFeatures
end

# ╔═╡ e202a63d-b119-47ac-bf70-6516fb29f423
begin
	DATA_PATH = joinpath(@__DIR__,"..","datasets","not_onco_combined_dataset")
	SAV_PATH = "$(DATA_PATH).sav"
	CSV_PATH = "$(DATA_PATH).csv"
end

# ╔═╡ f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
df = DataFrame(load(SAV_PATH))

# ╔═╡ 2bc52379-6bf6-4ccf-a418-3c5ca5adfaa0
CSV.write(CSV_PATH, df)

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
	
	dropped_variables = [ "mergeid", "hhid5", "hhid6", "hhid7", "mergeidp5", "mergeidp6", "mergeidp7", "coupleid5", "coupleid6", "coupleid7", "ph008d1", "ph054_", "ph080d1", "ph080d2", "ph080d3", "ph080d4", "ph080d5", "ph080d6", "ph080d7", "ph080d8", "ph080d9", "ph080d10", "ph080d11", "ph080d12", "ph080d13", "ph080d14", "ph080d15", "ph080d16", "ph080d17", "ph080d18", "ph080d19", "ph080d20", "ph080d21", "ph080d22", "ph080dot", "ph087d1", "ph087d2", "ph087d3", "ph087d4", "ph087d5", "ph087d6", "ph087d7", "ph088_", "ph089dno", "ph082_", "initial_euro_d", "euro_d", "ph006d21", "ph049d14", "ph049d15", "ph050_", "ph051_", "ph059d1", "ph059d2", "ph059d3", "ph059d4", "ph059d5", "ph059d6", "ph059d7", "ph059d8", "ph059d9", "ph059d10", "ph059dno", "ph059dot", "ph690d1", "ph690d2", "ph690d3", "ph690d4", "ph745_","ph009_1","ph009_2","ph009_3","ph009_4","ph009_5","ph009_6","ph009_10","ph009_11","ph009_12","ph009_13","ph009_14","ph009_15","ph009_16","ph009_18","ph009_19","ph009_20","ph009_other"]	
end

# ╔═╡ 804cee56-f1ba-4874-bc09-8b551e927c68
df2 = select(df, Not(dropped_variables))

# ╔═╡ 7f422edf-3657-4ad9-9468-1b6d4f8de46e
rename!(df2, Dict(Symbol(k) => Symbol(v) for (k, v) in attribute_names))

# ╔═╡ Cell order:
# ╠═494947b5-219b-43ad-b29f-56216b3dc639
# ╠═e202a63d-b119-47ac-bf70-6516fb29f423
# ╠═f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
# ╠═2bc52379-6bf6-4ccf-a418-3c5ca5adfaa0
# ╟─5d3d8370-1ad9-4721-99a9-d17768e8178a
# ╠═804cee56-f1ba-4874-bc09-8b551e927c68
# ╠═7f422edf-3657-4ad9-9468-1b6d4f8de46e
