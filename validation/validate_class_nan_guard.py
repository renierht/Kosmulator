"""Regression for the Model 9 CLASS numjac NaN crash (2026-10-15).
Run in Kosmulator_test from the repository root with its patched CLASS built.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Kosmulator_main import Class_run as C,rd_helpers as R
assert C.ensure_class_ready('SignSwitch_IDE_1',no_rebuild=True,announce=False)
base=['SignSwitch_IDE_1',.7,.307,.048,-1.03,.02,0.,0.,3.044,.06,3,0.]
references=[(-1.03,.02,149.60172547220958),(-.93,-.01,141.1245022408277),(-1.15,.04,154.56088199259085),(-1.,0.,144.09808966912607)]
rows=[]
for w,d,expected in references:
 p=base.copy();p[4]=w;p[5]=d;rd=R._rd_model_class_core(*p)
 assert abs(rd/expected-1)<1e-8,(rd,expected)
 rows.append(dict(w=w,delta=d,rd_before=expected,rd_after=rd,difference=rd-expected))
crash=['SignSwitch_IDE_1',.7416782117591887,.30705956939982576,.048,-1.06468083280595,-.18805327362650148,0.,0.,3.044,.06,3,0.]
former_point=[]
for repeat in range(3):
 R._rd_model_class_core.cache_clear()
 try:
  rd=R._rd_model_class_core(*crash)
  assert __import__("math").isfinite(rd) and rd>0
  former_point.append(rd)
 except Exception as e:
  assert 'Non-finite' in str(e),str(e)
  former_point.append('controlled CLASS failure')

# Verify subsequent good evaluations still work after repeated failures.
R._rd_model_class_core.cache_clear()
assert abs(R._rd_model_class_core(*base)/references[0][2]-1)<1e-8
result=dict(successful_points=rows,former_crash_outcomes=former_point,repeats=3,good_point_after_failure=True)
Path('validation/class_nan_guard_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
