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

# ╔═╡ Cell order:
# ╠═494947b5-219b-43ad-b29f-56216b3dc639
# ╠═e202a63d-b119-47ac-bf70-6516fb29f423
# ╠═f6b53ccf-85e2-40bd-bef4-6a329b1bf2d4
# ╠═2bc52379-6bf6-4ccf-a418-3c5ca5adfaa0
