using Documenter

push!(LOAD_PATH,"../src/")
makedocs(
    authors="Marcus Webb <marcusdavidwebb@gmail.com>",
    sitename = "classix",
    format = Documenter.HTML(),
    pages=["Home" => "index.md"],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/nla-group/classix.jl.git"
)
