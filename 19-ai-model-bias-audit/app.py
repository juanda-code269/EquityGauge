import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Model Equity Audit Lab",page_icon="⚖️",layout="wide")
st.title("Model Equity Audit Lab")
st.caption("Compare group-conditional metrics and trade-offs without reducing fairness to one number.")


@st.cache_data
def demo(n=5000):
    rng=np.random.default_rng(67);group=rng.choice(["Group A","Group B","Group C"],n,p=[.48,.34,.18]);x1=rng.normal(0,1,n)+np.select([group=="Group B",group=="Group C"],[.25,-.3],0);x2=rng.normal(0,1,n)
    opportunity=rng.normal(0,1,n)+np.select([group=="Group B",group=="Group C"],[-.2,-.45],0);latent=.9*x1+.55*x2+.4*opportunity+rng.normal(0,1,n);label=(latent>.25).astype(int)
    return pd.DataFrame({"signal_1":x1,"signal_2":x2,"context_measure":opportunity,"group_for_audit":group,"label":label})


def metrics(y,pred,group):
    rows=[]
    for g in sorted(pd.Series(group).astype(str).unique()):
        mask=pd.Series(group).astype(str).to_numpy()==g;tn,fp,fn,tp=confusion_matrix(np.asarray(y)[mask],np.asarray(pred)[mask],labels=[0,1]).ravel()
        rows.append({"Group":g,"N":mask.sum(),"Accuracy":(tp+tn)/mask.sum(),"Precision":tp/max(tp+fp,1),"Recall / TPR":tp/max(tp+fn,1),"False-positive rate":fp/max(fp+tn,1),"False-negative rate":fn/max(fn+tp,1),"Selection rate":(tp+fp)/mask.sum()})
    return pd.DataFrame(rows)


mode=st.sidebar.radio("Input mode",["Train demo models","Upload labels and predictions"])
if mode=="Upload labels and predictions":
    up=st.sidebar.file_uploader("Audit CSV",type="csv")
    if up is None:st.info("Upload a CSV containing label, group, and one or two prediction/probability columns.");st.stop()
    df=pd.read_csv(up);label=next((c for c in df if c.lower() in {"label","target","actual","y_true"}),None);group=next((c for c in df if c.lower() in {"group","sensitive_group","protected_group","group_for_audit"}),None);probs=[c for c in df if c.lower() in {"prediction","probability","score","model_a","model_b","y_pred"}]
    if not label or not group or not probs:st.error("Need label, group, and at least one prediction/probability column.");st.stop()
    y=df[label].astype(int).to_numpy();g=df[group].astype(str).to_numpy();available={c:df[c].astype(float).to_numpy() for c in probs}
else:
    df=demo();features=["signal_1","signal_2","context_measure"];train,test=train_test_split(np.arange(len(df)),test_size=.35,random_state=42,stratify=df.label);available={}
    for name,model in {"Logistic model":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),"Random forest":RandomForestClassifier(n_estimators=260,min_samples_leaf=8,random_state=42)}.items():
        model.fit(df.loc[train,features],df.label.iloc[train]);available[name]=model.predict_proba(df.loc[test,features])[:,1]
    y=df.label.iloc[test].to_numpy();g=df.group_for_audit.iloc[test].to_numpy()
model_name=st.sidebar.selectbox("Model",list(available));threshold=st.sidebar.slider("Decision threshold",.05,.95,.50,.01);score=available[model_name];pred=score>=threshold;table=metrics(y,pred,g)
tabs=st.tabs(["Group metrics","Disparities","Compare models","Threshold trade-offs","Interpretation"])
with tabs[0]:
    overall=pd.DataFrame({"Metric":["Accuracy","Precision","Recall","Selection rate"],"Overall":[accuracy_score(y,pred),precision_score(y,pred,zero_division=0),recall_score(y,pred),pred.mean()]});st.dataframe(overall.style.format({"Overall":"{:.1%}"}),width="stretch",hide_index=True)
    st.dataframe(table.style.format({c:"{:.1%}" for c in table.columns if c not in {"Group","N"}}),width="stretch",hide_index=True)
with tabs[1]:
    long=table.melt(id_vars=["Group","N"],var_name="Metric",value_name="Value");st.plotly_chart(px.bar(long,x="Metric",y="Value",color="Group",barmode="group",title=f"Metrics at threshold {threshold:.2f}"),width="stretch")
    metrics_cols=[c for c in table if c not in {"Group","N"}];gap=pd.DataFrame({"Metric":metrics_cols,"Max–min gap":[table[c].max()-table[c].min() for c in metrics_cols]});st.dataframe(gap.style.format({"Max–min gap":"{:.1%}"}),width="stretch",hide_index=True)
with tabs[2]:
    rows=[]
    for name,s in available.items():
        t=metrics(y,s>=threshold,g);rows.append({"Model":name,"Overall accuracy":accuracy_score(y,s>=threshold),"TPR gap":t["Recall / TPR"].max()-t["Recall / TPR"].min(),"FPR gap":t["False-positive rate"].max()-t["False-positive rate"].min(),"Selection-rate gap":t["Selection rate"].max()-t["Selection rate"].min()})
    st.dataframe(pd.DataFrame(rows).style.format({c:"{:.1%}" for c in rows[0] if c!="Model"}),width="stretch",hide_index=True)
    st.caption("Better overall performance and smaller gaps need not occur in the same model.")
with tabs[3]:
    rows=[]
    for t in np.linspace(.05,.95,37):
        m=metrics(y,score>=t,g);rows.append({"Threshold":t,"Accuracy":accuracy_score(y,score>=t),"TPR gap":m["Recall / TPR"].max()-m["Recall / TPR"].min(),"FPR gap":m["False-positive rate"].max()-m["False-positive rate"].min(),"Precision gap":m.Precision.max()-m.Precision.min()})
    sweep=pd.DataFrame(rows);st.plotly_chart(px.line(sweep,x="Threshold",y=["Accuracy","TPR gap","FPR gap","Precision gap"],title="Conflicting objectives across thresholds"),width="stretch")
with tabs[4]:
    st.markdown("""No single metric proves a model fair or unfair. Equal false-positive rates, equal true-positive rates, predictive parity, calibration, individual fairness, and equal selection rates answer different normative questions and can conflict—especially when observed label rates differ. Statistical disparity alone does not identify its cause.

Potential sources include sampling differences, historical inequality, measurement error, missing variables, label bias, and deployment feedback. Sensitive attributes appear here **for auditing**, not to justify decisions. Legitimate audits require domain context, affected-stakeholder input, privacy and legal review, uncertainty estimates, intersectional analysis, and examination of the surrounding decision process—not just model scores.""")

