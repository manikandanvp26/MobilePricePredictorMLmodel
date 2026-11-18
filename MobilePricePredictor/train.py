"""Train and compare models using the uploaded real-world dataset."""
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import model_core as core

def main():
    df=pd.read_csv(core.DATA_PATH).dropna(subset=[core.TARGET])
    X=df[core.FEATURES].copy(); y=df[core.TARGET].astype(float)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=core.RANDOM_STATE)
    cv=KFold(n_splits=5,shuffle=True,random_state=core.RANDOM_STATE)
    results={}; fitted={}
    for name in core.MODEL_NAMES:
        m=core.make_pipeline(name)
        scores=cross_val_score(m,Xtr,ytr,cv=cv,scoring='r2',n_jobs=1)
        m.fit(Xtr,ytr); pred=np.maximum(0,m.predict(Xte)); fitted[name]=m
        results[name]={'label':core.MODEL_LABELS[name],'cv_r2_mean':round(float(scores.mean()),4),'cv_r2_std':round(float(scores.std()),4),'test_r2':round(float(r2_score(yte,pred)),4),'test_mae_inr':round(float(mean_absolute_error(yte,pred)),2),'test_rmse_inr':round(float(mean_squared_error(yte,pred)**.5),2),'test_mape_pct':round(float(mean_absolute_percentage_error(yte,pred)*100),2),'within_10pct':round(float(np.mean(np.abs(pred-yte)/np.maximum(yte,1)<=.1)*100),1)}
    for name in core.MODEL_NAMES:
        resid_m=np.abs(yte.to_numpy()-np.maximum(0,fitted[name].predict(Xte)))
        results[name]['uncertainty_abs_90']=round(float(np.quantile(resid_m,.90)),2)
    best=max(results,key=lambda k:results[k]['cv_r2_mean'])
    final=core.make_pipeline(best).fit(X,y)
    pred_test=np.maximum(0,fitted[best].predict(Xte)); residual=np.abs(yte.to_numpy()-pred_test)
    uncertainty=float(np.quantile(residual,.90))
    joblib.dump({'model':final,'best_model':best,'features':core.FEATURES,'brands':sorted(df.brand.unique()),'uncertainty_abs':uncertainty},core.MODEL_PATH)
    perm=permutation_importance(fitted[best],Xte,yte,n_repeats=5,random_state=42,scoring='r2')
    imp=np.maximum(perm.importances_mean,0); total=float(imp.sum()) or 1
    importance=sorted([{'feature':f,'label':core.FEATURE_LABELS[f],'importance':round(float(v/total*100),1)} for f,v in zip(core.FEATURES,imp)],key=lambda x:-x['importance'])
    metrics={'best_model':best,'rows':len(df),'models':results,'feature_importance':importance,'uncertainty_abs_90':round(uncertainty,2),'dataset':'Mobile Price Prediction Datatset.csv','target':'INR'}
    core.METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Rows:',len(df),'Best:',core.MODEL_LABELS[best]); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
