"""Compare completed native exports; no likelihood recomputation or chain editing."""
from pathlib import Path
import json,csv
root=Path(__file__).resolve().parents[2];base=root/'Statistical_analysis_tables'
labels={'LCDM':'Model5_EOS_CC_v1_lcdm_iw_cc','Model4_fixed_DE':'Model4_EOS_CC_v2_model4_lambda_iw_cc','Model4_free_DE':'Model4_EOS_CC_v2_model4_free_de_burn3000_iw_cc'}
rows={}
for label,folder in labels.items():
 data=json.loads((base/folder/'model_comparison.json').read_text())
 assert len(data)==1
 group=next(iter(data.values()));assert len(group)==1
 rows[label]=next(iter(group.values()))
ref=rows['LCDM']; result=[]
for label,row in rows.items():
 for key in ['Dataset_signature','N','Data_counts','IC_convention']:
  assert row[key]==ref[key],(label,key)
 r={'Model':label,'burn':row['Burn'],'k':row['k'],'p_D':row['p_D']}
 for key in ['Chi_squared','AIC','AICc','DIC','BIC']:
  r[key]=row[key];r['delta_'+key]=row[key]-ref[key]
 result.append(r)
out=base/labels['Model4_free_DE']
(out/'comparison_with_LCDM_and_fixed_DE.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'comparison_with_LCDM_and_fixed_DE.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
print(json.dumps(result,indent=2))
