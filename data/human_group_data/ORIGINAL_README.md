# Scaffolding Cooperation in Human Groups with Deep Reinforcement Learning

Datasets and analysis scripts for studies conducted by
[McKee et al. (2022)](https://psyarxiv.com/cr58a).


## Contents

This repo contains four R analysis scripts to reproduce the results and plots
presented in McKee et al. (2022):

  * `baseline_analysis.R`: Analysis script for the baseline conditions.
  * `training_analysis.R`: Analysis script for agent training.
  * `evaluation_analysis.R`: Analysis script for the evaluation condition.
  * `validation_analysis.R`: Analysis script for the validation conditions.

It also contains fourteen datasets:

  * `baseline_cooperation_data.csv`: Node-level dataset for the baseline
  conditions, including player cooperation choices from each round.
  * `baseline_graph_struct_data.csv`: Edge-level dataset for the baseline
  conditions, including social planner recommendations from each round.
  * `baseline_group_outcomes_data.csv`: Graph-level dataset for the baseline
  conditions, including mean cooperation level and choice assortativity for each
  round.
  * `baseline_cooperation_glmer_eff_data.csv`: Dataset containing bootstrap
  confidence intervals computed for a generalized linear mixed-effects model
  analyzing cooperation choice in the baseline conditions.
  * `training_data.csv`: Dataset containing measurements of agent performance
  over training.
  * `evaluation_cooperation_data.csv`: Node-level dataset for the evaluation
  condition, including player cooperation choices from each round.
  * `evaluation_graph_struct_data.csv`: Edge-level dataset for the evaluation
  condition, including social planner recommendations from each round.
  * `evaluation_group_outcomes_data.csv`: Graph-level dataset for the evaluation
  condition, including mean cooperation level and choice assortativity for each
  round.
  * `evaluation_cooperation_glmer_eff_data.csv`: Dataset containing bootstrap
  confidence intervals computed for a generalized linear mixed-effects model
  analyzing cooperation choice in the baseline and evaluation conditions.
  * `evaluation_node_representation_data.csv`: Dataset containing node-related
  variables, including the GraphNet node representations and the core-periphery
  classification for each player through the game.
  * `validation_cooperation_data.csv`: Node-level dataset for the validation
  conditions, including player cooperation choices from each round.
  * `validation_graph_struct_data.csv`: Edge-level dataset for the validation
  conditions, including social planner recommendations from each round.
  * `validation_group_outcomes_data.csv`: Graph-level dataset for the validation
  conditions, including mean cooperation level and choice assortativity for each
  round.
  * `validation_cooperation_glmer_eff_data.csv`: Dataset containing bootstrap
  confidence intervals computed for a generalized linear mixed-effects model
  analyzing cooperation choice in the baseline, evaluation, and validation
  conditions.

Finally, it contains a `renv.lock` file capturing the state of the R environment
used to generate the results in McKee et al. (2022). Specifically, the lockfile
details the versions of R and all library packages loaded in the R environment.

## Installing non-default packages

The R analysis scripts for this project use the following non-default packages:
`car`, `emmeans`, `Hmisc`, `lme4`, `pROC`, and `tidyverse`. Before running any
scripts, install these packages by running the following command in an R
session:

```
install.packages(c("car", "emmeans", "Hmisc", "lme4", "pROC", "tidyverse"))
```

To check for version differences between your R environment and the environment
used to generate the results in McKee et al. (2022), you can separately install
the `renv` package. Then, in an R session with the working directory set to the
folder containing the scripts from the repo, run the following command:

```
renv::status()
```

## Running a script

In an R session, set the working directory to the folder containing the script
and datasets associated with one of the studies. The script can then be run with
`source("[script_name].R", echo = T)`. This will automatically print each
expression in the script in order, as well as any output resulting from
evaluating the expression (e.g., the results from a statistical test).

For example, to reproduce the results and plots for the baseline analysis, run
the following command:

```
source("baseline_analysis.R", echo = T)
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
