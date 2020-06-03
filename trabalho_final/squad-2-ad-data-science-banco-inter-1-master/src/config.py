# algorithms parameters


class Configure:
    def __init__(self):
        self.pf1_folder = '../data/estaticos_portfolio1.csv'
        self.pf2_folder = '../data/estaticos_portfolio2.csv'
        self.pf3_folder = '../data/estaticos_portfolio3.csv'
        self.mkt_folder = '../data/estaticos_market.csv'
        self.pre_processing_params = {}
        self.feature_selection_params = {}
        self.model_params = {}
        self.evaluation_params = {}

    def set_pre_processing_params(self):
        manual_encoding = {'de_saude_tributaria': {
                             'VERDE': 0,
                             'AZUL': 1,
                             'AMARELO': 2,
                             'CINZA': 3,
                             'LARANJA': 4,
                             'VERMELHO': 5},
                            'de_nivel_atividade': {
                             'MUITO BAIXA': 0,
                             'BAIXA': 1,
                             'MEDIA': 2,
                             'ALTA': 3}
                            }
        self.pre_processing_params['manual_encoding'] = manual_encoding

    def set_fs_params(self):
        d = {'threshold': 0.7,
             'EDA': ['fl_epp',
                     'qt_socios_pf',
                     'idade_maxima_socios',
                     'idade_minima_socios',
                     'qt_socios_st_regular',
                     'qt_socios_masculino',
                     'de_saude_rescencia',
                     'de_faixa_faturamento_estimado',
                     'de_faixa_faturamento_estimado_grupo',
                     'vl_faturamento_estimado_grupo_aux',
                     'idade_emp_cat',
                     'de_saude_rescencia',
                     'fl_me',
                     'fl_email',
                     'nu_meses_rescencia',
                     'fl_st_especial',
                     #'sg_uf',
                     'nm_meso_regiao',
                     'sg_uf_matriz',
                     'nm_micro_regiao',
                     'nm_segmento',
                     'nm_divisao',
                     'de_natureza_juridica',
                     #'setor'
                     'de_ramo']
             }
        self.feature_selection_params = d

