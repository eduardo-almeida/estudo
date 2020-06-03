# [`recommender_system_squad_1`]
> Simple project description.

This model is a `recommendation_system` that `suggest new customers` based on a database with more than 460 000 clients. The notebook `Recommendation_system.ipynb`  in `recommender_system_squad_1 folder` is running the recommendation system and use `clientes2_merge.csv` as example.

## Stakeholders
> Describe the people involved in this project

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | [`bruno-manoel-dbki`]            | [`brunomanoel@dobrovolski.com.br`] |
| Data Scientist       | Author              | [`Fernando Santana Pacheco`] | [`fspacheco@gmail.com`]   |

## Usage
> Describe how to reproduce your model

Starts cloning the project from the analytics Models repo.
```
git clone https://github.com/<@github_username>/recommender_system_squad_1.git
cd recommender_system_squad_1
```

### Analysis
In folder analysis you'll find all steps We need to get our reccomendation system. From Exploratory analysis to Clusterization. 
In EDA folder We generate many results with pandas profilling. The HTML files help us to select features to be deleted. 
In Clustering folder We test some clusterization methods with reduced features, looking for a visual result.
Back to analysis folder we can see the `csv` data files and  a `csv` with selected features to exclude `exclude_variables_generated_from_profiling.csv`. This file is used by `Save full market database reduced columns.ipynb` that is our base result to start to build the system.

### recommender_system_squad_1

This folder contains the final notebook 
`Recommendation_system.ipynb`

## Final Report (to be filled once the project is done)

### Model Frequency

Our system need about 4 minutes to run. Since the feather file is downloaded.

### Model updating

The update could came with new feather file, it contains the leads database


## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Explain you folder strucure

* [analysis](./analysis/): contains notebooks of data and modeling experimentation.
* [tests](./tests/): contains files used for unit tests.
* [recommender_system_squad_1](./recommender_system_squad_1/): main Python package with source of the model.
