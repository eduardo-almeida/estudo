import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
sns.set(style='darkgrid')
import warnings
warnings.filterwarnings('ignore')


# Carregando a base
market = pd.read_csv('estaticos_market.csv')
port1 = pd.read_csv('estaticos_portfolio1.csv')
port2 = pd.read_csv('estaticos_portfolio2.csv')
port3 = pd.read_csv('estaticos_portfolio3.csv')

# Cria indicadoras
market['port1'] = market.id.isin(port1.id)
market['port2'] = market.id.isin(port2.id)
market['port3'] = market.id.isin(port3.id)
market['idc_market'] = 1

# Dropa coluna unnamed
market.drop('Unnamed: 0', axis=1, inplace=True)

# Seleciona apenas os dados com menos de 50% de nulos 
market = market.loc[:, market.isnull().mean() < 0.5]

#Criando um dataframe auxliar para analisar a consistencia das variaveis
cons = pd.DataFrame({'colunas' : market.columns,
                    'tipo': market.dtypes,
                    'missing' : market.isna().sum(),
                    'size' : market.shape[0],
                  'unicos': market.nunique()})
cons['percentual'] = round(cons['missing'] / cons['size'],2)

natureza_privado = ['SOCIEDADE EMPRESARIA LIMITADA', 'EMPRESARIO INDIVIDUAL', 'COOPERATIVA', 'ASSOCIACAO PRIVADA', 
'ENTIDADE SINDICAL', 'CONSORCIO DE SOCIEDADES', 'SOCIEDADE DE ECONOMIA MISTA', 'CONDOMINIO EDILICIO', 'SOCIEDADE ANONIMA ABERTA',
'EMPRESA INDIVIDUAL DE RESPONSABILIDADE LIMITADA DE NATUREZA EMPRESARIA', 'EMPRESA INDIVIDUAL IMOBILIARIA',
'SOCIEDADE SIMPLES LIMITADA', 'FUNDACAO PRIVADA', 'SOCIEDADE ANONIMA FECHADA',  'ORGANIZACAO RELIGIOSA',
'SOCIEDADE UNIPESSOAL DE ADVOCACIA', 'SOCIEDADE SIMPLES PURA', 'SOCIEDADE EMPRESARIA EM NOME COLETIVO',
'EMPRESA INDIVIDUAL DE RESPONSABILIDADE LIMITADA DE NATUREZA SIMPLES', 'SERVICO NOTARIAL E REGISTRAL CARTORIO', 
'ORGANIZACAO SOCIAL OS', 'CONSORCIO PUBDE DIREITO PUB ASS PUB', 'SERVICO SOCIAL AUTONOMO', 'SOCIEDADE EM CONTA DE PARTICIPACAO',
'GRUPO DE SOCIEDADES', 'SOCIEDADE MERCANTIL DE CAPITAL E INDUSTRIA', 'SOCIEDADE EMPRESARIA EM COMANDITA POR ACOES',
'ESTABELECIMENTO NO BRASIL DE SOCIEDADE ESTRANGEIRA', 'ENTIDADE DE MEDIACAO E ARBITRAGEM', 'COMUNIDADE INDIGENA',
'ESTABELECIMENTO NO BRASIL DE FUNDACAO OU ASSOCIACAO ESTRANGEIRAS','CLUBE FUNDO DE INVESTIMENTO', 'REPRESENTACAO DIPLOMATICA ESTRANGEIRA',
'SOCIEDADE SIMPLES EM COMANDITA SIMPLES', 'FRENTE PLEBISCITARIA OU REFERENDARIA', 'SOCIEDADE EMPRESARIA EM COMANDITA SIMPLES',
'CONSORCIO DE EMPREGADORES', 'SOCIEDADE SIMPLES EM NOME COLETIVO', 'FUNDO PRIVADO']

natureza_publico = ['ORGAO PUBLICO DO PODER EXECUTIVO FEDERAL', 'MUNICIPIO', 'ORGAO DE DIRECAO LOCAL DE PARTIDO POLITICO',
'CANDIDATO A CARGO POLITICO ELETIVO', 'ORGAO PUBLICO DO PODER EXECUTIVO ESTADUAL OU DO DISTRITO FEDERAL',
'ORGAO PUBLICO DO PODER EXECUTIVO MUNICIPAL', 'AUTARQUIA FEDERAL', 'ORGAO PUBLICO DO PODER JUDICIARIO ESTADUAL',
'EMPRESA PUBLICA', 'AUTARQUIA MUNICIPAL', 'FUNDO PUBLICO', 'AUTARQUIA ESTADUAL OU DO DISTRITO FEDERAL',
'FUNDACAO PUBLICA DE DIREITO PUB MUNICIPAL', 'ORGAO PUBLICO DO PODER LEGISLATIVO MUNICIPAL', 'ORGAO DE DIRECAO REGIONAL DE PARTIDO POLITICO',
'FUNDACAO PUBLICA DE DIREITO PRIVADO MUNICIPAL', 'ORGAO PUBLICO AUTONOMO MUNICIPAL', 'ORGAO PUBLICO DO PODER JUDICIARIO FEDERAL',
'FUNDACAO PUBLICA DE DIREITO PUBFEDERAL', 'ORGAO PUBLICO AUTONOMO ESTADUAL OU DO DISTRITO FEDERAL', 'FUNDACAO PUB DE DIREITO PUB EST OU DO DF',
'ORGAO PUBLICO DO PODER LEGISLATIVO ESTADUAL OU DO DISTRITO FEDERAL', 'FUNDACAO PUBLICA DE DIREITO PRIVADO FEFERAL',
'CONSORCIO PUBLICO DE DIREITO PRIVADO', 'ESTADO OU DISTRITO FEDERAL', 'ORGAO PUBLICO DO PODER LEGISLATIVO FEDERAL',
'ORGAO DE DIRECAO NACIONAL DE PARTIDO POLITICO']

market['setor_privado'] = np.where(market['de_natureza_juridica'].isin(natureza_privado), True, False)
market['fl_rm'] = market['fl_rm'].map({'SIM': True, 'NAO': False})
clientes_mei = pd.get_dummies(market['fl_mei'], 'col1', 'col2')
market['cli_mei'] = clientes_mei.col1col2True  # Adicionando uma nova coluna no market
cliente_simples = pd.get_dummies(market['fl_optante_simples'], 'col1', 'col2')
market['cli_simples'] = cliente_simples.col1col2True
variacao_faixa = {'ATE R$ 81.000,00': 1, 
                  'DE R$ 81.000,01 A R$ 360.000,00' : 2,
                  'DE R$ 360.000,01 A R$ 1.500.000,00' : 3, 
                  'DE R$ 1.500.000,01 A R$ 4.800.000,00' : 4, 
                  'DE R$ 4.800.000,01 A R$ 10.000.000,00' : 5,
                  'DE R$ 10.000.000,01 A R$ 30.000.000,00' : 6,
                  'DE R$ 30.000.000,01 A R$ 100.000.000,00' : 7,
                  'DE R$ 100.000.000,01 A R$ 300.000.000,00': 8, 
                  'DE R$ 300.000.000,01 A R$ 500.000.000,00' : 9, 
                  'DE R$ 500.000.000,01 A 1 BILHAO DE REAIS' : 10, 
                  'ACIMA DE 1 BILHAO DE REAIS' : 11} 

market['faturamento_cat'] = market['de_faixa_faturamento_estimado_grupo'].map(variacao_faixa)
market['faturamento_cat'] = np.where(market['de_faixa_faturamento_estimado_grupo'].isnull() & market['fl_mei'] == True, 1, market['faturamento_cat'])
market.faturamento_cat = np.where(market['faturamento_cat'].isnull() & market['fl_optante_simei'] == True, 1, market.faturamento_cat)
market.faturamento_cat = np.where(market['faturamento_cat'].isnull() & market['fl_optante_simples'] == True, 2, market.faturamento_cat)
market.faturamento_cat = np.where(market['faturamento_cat'].isnull() & market['fl_me'] == True, 2, market.faturamento_cat)
market['faturamento_cat'] = np.where(market['faturamento_cat'].isnull(), 1.5, market['faturamento_cat'])
market['faturamento_cat'] = np.where(market['faturamento_cat'].isnull(), 1.5, market['faturamento_cat'])
bins = [0, 1, 2, 5, 9, 65,  np.inf]
names = ['0-1','1-2', '2-5', '5-9', '9-65', '65+']
market['qt_filiais_range'] = pd.cut(market['qt_filiais'], bins, labels=names, include_lowest=True)
operacao = market.groupby(by='de_nivel_atividade')['port1', 'port2', 'port3', 'idc_market'].sum().reset_index()
operacao_alta = operacao[operacao['de_nivel_atividade'] == 'ALTA']
market['setor'] = np.where(market['setor'].isnull(), 'OUTROS', market['setor'])
market['nm_segmento'] = np.where(market['nm_segmento'].isnull(), 'OUTROS', market['nm_segmento'])
market['nao_regular'] = market['qt_socios'] - market['qt_socios_st_regular']