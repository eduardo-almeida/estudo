import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.style.use('seaborn-whitegrid')


def calc_auroc(df, mean1, feature, target):
    
    lst = []
    
    df[feature] = df[feature].fillna(0)

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Fator
                    val,                                                            # Categoria
                    df[df[feature] == val].mean()[mean1]*100,                       # Media
                    df[df[feature] == val].count()[feature],                        # Total
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Bons (DEFAULT == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Maus (DEFAULT == 1)

    data = pd.DataFrame(lst, columns=['Fator', 'Percentil_Prob', 'Prob_Media', 'Total', 'Bons', 'Maus'])

    data['Total (%)'] = data['Total'] / data['Total'].sum()
    data['Inadimpl.(%)'] = data['Maus'] / data['Total']
    data['Distr. Bons(%)'] = (data['Total'] - data['Maus']) / (data['Total'].sum() - data['Maus'].sum())
    data['Distr. Maus(%)'] = data['Maus'] / data['Maus'].sum()
    
    data = data.sort_values(by=['Fator', 'Percentil_Prob'], ascending=[True, True])
    data.index = range(len(data.index))

    data[['Total (%)','Inadimpl.(%)','Distr. Bons(%)','Distr. Maus(%)']] = data[['Total (%)','Inadimpl.(%)','Distr. Bons(%)','Distr. Maus(%)']] * 100
    
     
    j=0

    for i in range(data['Percentil_Prob'].nunique()):
        if j == 0:
            data.at[j,'Distr. Acum. Bons(%)'] = data.at[j,'Distr. Bons(%)']
            data.at[j,'Distr. Acum. Maus(%)'] = data.at[j,'Distr. Maus(%)']
            data.at[j,'AUROC'] = (data.at[j,'Distr. Acum. Maus(%)']/100)*(2-(data.at[j,'Distr. Acum. Bons(%)']/100))/2
        else:
            data.at[j,'Distr. Acum. Bons(%)'] = data.at[j-1,'Distr. Acum. Bons(%)']+data.at[j,'Distr. Bons(%)']
            data.at[j,'Distr. Acum. Maus(%)'] = data.at[j-1,'Distr. Acum. Maus(%)']+data.at[j,'Distr. Maus(%)']
            data.at[j,'AUROC'] = ((data.at[j,'Distr. Acum. Maus(%)']/100)-(data.at[j-1,'Distr. Acum. Maus(%)']/100))*(2-(data.at[j,'Distr. Acum. Bons(%)']/100)-(data.at[j-1,'Distr. Acum. Bons(%)']/100))/2

        data['KS (%)'] = abs(data['Distr. Acum. Bons(%)'] - data['Distr. Acum. Maus(%)'])
        j = j+1

    ks = round(data['KS (%)'].max(),2)
    #print('KS '+ feature +' = {:.2f}'.format(ks)) 

    AUROC = round(data['AUROC'].sum(),4)
    #print('AUROC '+ feature +' = {:.2f}'.format(AUROC))
    
 
    return ks, AUROC, data


#-------------------------------------------------------------------------
    

def grafico_eilert(indice,niter,metodo):
    
    DIF = list(range(len(indice)))
            
    for i in range(len(indice)):
        if i == 0: 
            DIF[i] = round(indice[i],4)
        else:
            DIF[i] = round(indice[i]-indice[i-1],4)

    fig, ax = plt.subplots(figsize=(12,5))
    ax2 = ax.twinx()    
    
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar(niter,DIF,alpha=0.8,width=0.8)
    ax.set_title(metodo + ": Gain per iteration",weight='bold',size=18)
    ax.set_ylabel("Gain",weight='bold',size=15)
    ax.set_xlabel("Iteration",weight='bold',size=15)

    
    for x,y in zip(niter,DIF):
        if metodo == "AUROC":
            label = "{:.4f}".format(y)
        else:
            label = "{:.2f}".format(y)
    
        ax.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
    
        
    ax2.grid(False)
    ax2.plot(niter,indice,'-b',label=metodo,linewidth=4, marker="D", markersize=10)
    ax2.set_xlabel("Iteration",weight='bold',size=15)
    ax2.set_ylabel(metodo,weight='bold',size=15)
    ax2.legend(fontsize=12,loc=5)
        
        
    for x,y in zip(niter,indice):
        if metodo == "AUROC":
            label = "{:.4f}".format(y)
        else:
            label = "{:.2f}".format(y)
    
        ax2.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
            
    plt.show()


def metodo_eilert(X, y, classificador, nit, metodo):
    
    var_char = X.columns[X.dtypes=='object']
    var_num  = X.columns[X.dtypes!='object']
    
    
    for var in var_char:
        X[var]=X[var].astype('category').cat.codes.astype('category')
    
    
    for var in var_num:
        X[var]=round(X[var].rank(pct=True)*100,0).astype('category')


    y = pd.DataFrame(y)

    y[0] = y[0].astype('category').cat.codes
    y = y[0].values
    
    selecao = []
    Eilert = pd.DataFrame()
            
    for i in range(nit):
        
        Lista = []
        Z = X.drop(columns=selecao)    
        
        for var in list(Z.columns):
            vartemp = selecao 
            vartemp = vartemp + [var]
            datatemp = pd.DataFrame(X[vartemp])
            classificador.fit(datatemp,y)
            W = pd.DataFrame(X[vartemp])
            W["Target"] = y
            W["Prob"] = classificador.predict_proba(datatemp)[:,1]
            W["Percentil"] = 100-round(W.Prob.rank(pct=True)*100,0)
            ks, AUROC, data = calc_auroc(W, "Prob", "Percentil", "Target")
            Lista.append([i+1,var,ks,AUROC])
            
        Lista = pd.DataFrame(Lista, columns=['Iteration','Variavel', 'KS', 'AUROC'])        
        Lista = Lista.sort_values(by=metodo, ascending=False).reset_index(drop=True)
        
        selecao = selecao + [Lista["Variavel"][0]]
        Eilert = Eilert.append(Lista[:1]).reset_index(drop=True)
        
    #grafico_eilert(Eilert[metodo], Eilert["Iteration"], metodo)
   
    
    return Eilert


'''


import scorecardpy as sc

y = pd.DataFrame(y,columns=['y'])

data = pd.concat([X,y],axis=1)

# woe binning ------
bins = sc.woebin(data, y="y")

# converting train and test into woe values
data = sc.woebin_ply(data, bins)



'''