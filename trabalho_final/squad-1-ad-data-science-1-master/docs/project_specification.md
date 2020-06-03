# Project specification - [`recommender_system_squad_1`]
> This document contains a data-science oriented description of the project. It orients the data science project towards data and technology. It serves the purposes of

* outlining a solution,
* defining the scope of the project,
* defining measurable success metrics,
* defining interfaces to the production environment,
* gathering information regarding relevant datasets and features to the problem,
* upstream communication and agreement on requisites.


## Checklist
> Mark which tasks have been performed.

- [ ] **Summary**: all major specs are defined including success criteria, scope, etc.
- [ ] **Output specification**: the output format, metadata and indexation have been defined and agreed upon.
- [ ] **Solution architecture**: a viable and minimal architecture for the final solution has been provided and agreed upon.
- [ ] **Limitations and risks**: a list with the main limitations and risks associated to the project has been provided.
- [ ] **Related resources**: lists with related datasets, features and past projects have been given.
- [ ] **Peer-review**: you discussed and brainstormed with colleagues the outlined specifications.
- [ ] **Acceptance**: these specifications have been accepted by the Data and Product directors. State names and date.

## Summary
> The table below summarizes the key requirements for the project.

| problem type              | target population | entity | N_target | min_coverage | N_labeled | sucess_metrics | updt_freq |
|---------------------------|-------------------|--------|----------|--------------|-----------|----------------|-----------|
| recommendation system | business       | CNPJ   | 30M      | 80%          | NA        | NA             |  sporadic  |


### Objective
> Provide a short (max 3-line) description  of the project's objective.

"This project aims at developing a recommendation system to recommend leads for a company based on a list of current clients.

### Target population
> More detailed description of the population to which the model should apply. Include any relevant characteristics.

| Entity | Region | Type        | Size | Status | Sector   | N_target |
|--------|--------|-------------|------|--------|----------|----------|
| CNPJ   | Brasil | companues | any  | active | all | 30M      |


#### Subsetting
> Also provide a list of sub-setting variables and how they related to the target population.

| Subsetting variable  | Selection rule                               |
|----------------------|----------------------------------------------|
| `cd_ramo_atividade`  | belongs to ['5611201', '5611202', '5611202'] |
| `situacao_cadastral` | 'Ativa'                                      |

### Output specification
> Describe how the output of the model will be delivered, including its domain and metadata.

The output is in principal notebook, showing a map marked with current clients and new leads.

#### Metadata
> Your model's metada should be provided in a machine-readable format (e.g. a json file) and include the following items:

* a brief description: this model predicts the type of a restaurant cuisine
* a measure of accuracy applicable for use of the model in other setups (if applicable): standard deviation, accuracy, error matrix.
* model version
* author
* date created
* link to training data

### Problem type
> Describe to which Data Science problem type(s) this project relates to with a brief motivation.

"Since the objective is to recommend new leads to a client, this problem is a recomemndation problem. It is also unsupervised since no observed data is available."


### Limitations and risks
> Provide a list with the main limitations and the associated risks for this project. (lile are supposed to be a well-educated guess)

| Limitation                              | Likelihood | Loss                               | Contingency                        |
|-----------------------------------------|------------|------------------------------------|------------------------------------|
| IDs not contained in the base           | 100%       | system wont suggest any lead  | Insert this clients and their data in the base. |
| High number of clients with missing information in database | 50%        | The quality of suggestions could not be interesting | Update more informaiton about clients in the database.            |

