# [`aceleradev-data-science-7`]
> Simple project description.

O objetivo deste produto é fornecer um serviço automatizado que recomenda leads para um usuário dado sua atual lista de clientes (Portfólio).

## Stakeholders
> Describe the people involved in this project

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | [`Claudio Silva`]        | [`andsil@gmail.com`] |
| Data Scientist       | Author             	| [`Wagner`] 		   | [`wagnerfs1989@hotmail.com`]   |

## Usage
> Describe how to reproduce your model

Algumas empresas gostariam de saber quem são as demais empresas em um determinado mercado (população) que tem maior probabilidade se tornarem seus próximos clientes. Ou seja, a sua solução deve encontrar no mercado quem são os leads mais aderentes dado as características dos clientes presentes no portfólio do usuário.

Além disso, sua solução deve ser agnóstica ao usuário. Qualquer usuário com uma lista de clientes que queira explorar esse mercado pode extrair valor do serviço.

Para o desafio, deverão ser consideradas as seguintes bases:

Mercado: Base com informações sobre as empresas do Mercado a ser considerado. Portfolio 1: Ids dos clientes da empresa 1 Portfolio 2: Ids dos clientes da empresa 2 Portfolio 3: Ids dos clientes da empresa 3

Obs: todas as empresas(ids) dos portfolios estão contidos no Mercado(base de população).

Makefile commands can be accessed using `make help`.

Make sure that **docker** is installed.

Clone the project from the analytics Models repo.
```
git clone https://github.com/<@github_username>/projeto_final.git
cd projeto_final
```


## Final Report (to be filled once the project is done)

### Model Frequency

> Describe the interval frequency and estimated total time to run

### Model updating

> Describe how your model may be updated in the future

### Maintenance

> Describe how your model may be maintained in the future

### Minimum viable product

> Describe a minimum configuration that would be able to create a minimum viable product.

### Early adopters

> Describe any potential paying users for this product if it was available today. Also state a point of contact for each of them.

## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Explain you folder strucure

* [docs](./docs): contains documentation of the project
* [analysis](./analysis/): contains notebooks of data and modeling experimentation.
* [tests](./tests/): contains files used for unit tests.
* [projeto_final](./projeto_final/): main Python package with source of the model.
