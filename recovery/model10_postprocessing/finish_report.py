from pathlib import Path
import json,re
root=Path(__file__).resolve().parents[2];warm=root.parent/'Kosmulator_warm_dm'
source=root/'Statistical_analysis_tables/LCDM_CPL_Model9_Model10_comparison'
r=json.loads((source/'model_comparison.json').read_text())
order=['LCDM_v','w0waCDM','SignSwitch_IDE_1','SignSwitch_IDE_2']
labels=['LCDM','CPL','IDE Model 9','IDE Model 10']
text='# Four-model comparison\n\nDESI DR2 + Pantheon+ with SH0ES; N=1670; fixed burn=1000; native polished likelihood maxima and posterior-mean DIC. All differences are model minus LCDM.\n\n| Model | chi2 | AIC | AICc | DIC | BIC |\n|---|---:|---:|---:|---:|---:|\n'
for name,label in zip(order,labels):
 row=next(iter(r[name].values()));text+='| '+label+' | '+' | '.join(f'{row[k]:.3f}' for k in ['Chi_squared','AIC','AICc','DIC','BIC'])+' |\n'
text+='\n| Model | Delta AIC | Delta AICc | Delta DIC | Delta BIC | Approx. LRT sigma |\n|---|---:|---:|---:|---:|---:|\n'
for name,label in zip(order,labels):
 row=next(iter(r[name].values()));text+='| '+label+' | '+' | '.join(f'{row[k]:.3f}' for k in ['dAIC','dAICc','dDIC','dBIC'])+' | '+(f"{row['LRT_sigma']:.3f}" if row.get('LRT_sigma') is not None else 'unavailable / reference')+' |\n'
text+='\n## Model 10 recovery\n\nThe original failure was reproduced in the best-fit plotting stage: the artificial vector made from all marginal lower endpoints is invalid in the CLASS Model 10 background. The median vector is valid. The native plot now omits this endpoint band and labels the omission; no fallback rd or likelihood modification was introduced. A proper joint posterior predictive band would require a separate calculation. The completed corner plot and original chain were preserved. The expensive 6000-draw rd corner calculation was not repeated. The recovered aligned parameter table lists sampled parameters; the existing corner plot retains the derived rd summary.\n\nNative statistical analysis was rerun with the original paper polishing options and all retained samples. The other three native results were reused. Four-model exports include separate posterior-preference and likelihood-ratio eligibility reports.\n\n'
row=next(iter(r['SignSwitch_IDE_2'].values()));text+='Model 10 LRT status: '+row.get('LRT_status','unknown')+'.\n\n';text+='Model 10 optimizer runs: '+str([(x.get('success'),x.get('chi2'),x.get('message')) for x in row['Optimizer_runs']])+'.\n\n'
text+='Validation: 79 regression tests passed, including checks that invalid endpoints omit the band, while invalid central values and programming errors still propagate. The saved chains, priors and sampler configuration were not modified. Existing CLASS-domain limitations and independent-ensemble caveats remain.\n'
for folder in [source,warm/'Statistical_analysis_tables/LCDM_CPL_Model9_Model10_comparison',root/'Statistical_analysis_tables/Model10_unrestricted_v1_iw']:
 (folder/'COMPARISON.md').write_text(text)
 for name in order:
  row=next(iter(r[name].values()));(folder/name/'interpretations_summary.txt').write_text('Reference: LCDM_v. Negative delta IC favors this model.\n'+ '\n'.join(f'{k}: {row.get(k)}' for k in ['dAIC','dAICc','dDIC','dBIC','LRT_sigma','LRT_status'])+'\n')
p=root/'Statistical_analysis_tables/Model10_unrestricted_v1_iw/SignSwitch_IDE_2/aligned_table_pretty.txt';p.write_text(re.sub(r'\x1b\[[0-9;]*m','',p.read_text()))
(root/'recovery/model10_postprocessing/COMPLETED.md').write_text(text)
print(text)
