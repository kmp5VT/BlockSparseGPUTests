include("settings.jl")

makedocs(; sitename=sitename, settings...)

# If ENV["GITHUB_EVENT_NAME"] == "workflow_dispatch"
# it indicates the Documenter build was launched manually,
# by a GitHub action run through the GitHub website.
# As of Dec 2022, Documenter does not build the dev branch
# in this case, so change the value to "push" to fix:
if get(ENV, "GITHUB_EVENT_NAME", nothing) == "workflow_dispatch"
  ENV["GITHUB_EVENT_NAME"] = "push"
end

deploydocs(;
  repo="https://github.com/kmp5VT/BlockSparseGPUTests",
  devbranch="main",
  push_preview=true,
  deploy_config=Documenter.GitHubActions(),
)
