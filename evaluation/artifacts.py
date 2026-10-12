import pandas as pd, re, json, sys
d=sys.argv[1] if len(sys.argv) > 1 else "data/raw"
f=pd.read_csv(f'{d}/Fake.csv');t=pd.read_csv(f'{d}/True.csv')
f['label']=1;t['label']=0
df=pd.concat([f,t],ignore_index=True)
pats={'(Reuters) tag':r'\(Reuters\)','Dateline "CITY (Reuters) -"':r'^[A-Z][A-Za-z .,/\'-]{1,60}\(Reuters\) -',
'"Featured image"':r'Featured image','"21st Century Wire"':r'21st Century Wire','URL (http/www)':r'https?://|www\.',
'pic.twitter.com':r'pic\.twitter\.com','@handle':r'@\w+','"Getty Images"':r'Getty',
'"via" photo credit':r'(?i)image via|photo by','Title fully upper-case word (>=1)':None}
rows=[]
for k,p in pats.items():
    if p is None:
        s=df.title.str.contains(r'\b[A-Z]{4,}\b')
    else:
        s=df.text.str.contains(p,regex=True)
    rows.append({'marker':k,'fake_pct':round(100*s[df.label==1].mean(),2),'real_pct':round(100*s[df.label==0].mean(),2)})
r=pd.DataFrame(rows);print(r.to_string(index=False))
# empty texts
print('empty/whitespace text fake',(f.text.str.strip()=='').sum(),'real',(t.text.str.strip()=='').sum())
# duplicates
full=(df.title.astype(str)+' '+df.text.astype(str)).str.strip().str.lower()
print('exact duplicate rows (title+text):',full.duplicated().sum())
print('duplicate text (non-empty):',df.text[df.text.str.strip()!=''].str.strip().duplicated().sum())
print('dup titles',df.title.str.strip().str.lower().duplicated().sum())
# cross-label duplicates
g=full.groupby(full).apply(lambda x: df.label[x.index].nunique())
print('texts appearing with both labels',(g>1).sum())
# dates
for name,x in [('fake',f),('real',t)]:
    dd=pd.to_datetime(x.date.str.strip(),errors='coerce',format='mixed')
    print(name,'date min',dd.min(),'max',dd.max(),'unparsed',dd.isna().sum())
print('subjects fake',f.subject.unique(),'real',t.subject.unique())
print('mean words fake',f.text.str.split().str.len().mean().round(1),'real',t.text.str.split().str.len().mean().round(1))
r.to_csv('evaluation/results/artifacts.csv',index=False)
