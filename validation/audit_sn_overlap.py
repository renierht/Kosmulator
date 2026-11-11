"""Exact local identifier matches: positive matches flag overlap, absence proves nothing."""
import json
from pathlib import Path
from itertools import combinations
import pandas as pd
p=pd.read_csv('Observations/PantheonP.dat',sep=r'\s+')
y=pd.read_csv('Observations/DESY5.dat')
d=pd.read_csv('Observations/DES_Dovekie/DES-Dovekie_HD.csv',sep=r'\s+',comment='#')
samples={'PantheonP':p[p.zHD>.01], 'PantheonPS':p[(p.zHD>.01)|(p.IS_CALIBRATOR>0)],'DESY5':y,'DES_Dovekie':d}
report=[]
for (name,a),(other,b) in combinations(samples.items(),2):
    ca='#CID' if '#CID' in a else 'CID'; cb='#CID' if '#CID' in b else 'CID'
    keys=lambda frame,col:set(zip(frame[col].astype(str).str.strip().str.lower(),frame.IDSURVEY.astype(int)))
    same=keys(a,ca)&keys(b,cb)
    ids=set(a[ca].astype(str).str.lower())&set(b[cb].astype(str).str.lower())
    report.append(dict(first=name,second=other,exact_CID_and_survey_matches=len(same),exact_CID_matches=len(ids)))
Path('validation/sn_overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
