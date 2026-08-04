#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/"compute-center"
digest=os.environ.get("SAGEMATH_IMAGE_DIGEST","").strip()
version=os.environ.get("SAGEMATH_VERSION","").strip()
if not digest.startswith("sagemath/sagemath@sha256:"):
    raise SystemExit("SAGEMATH_IMAGE_DIGEST must be an exact RepoDigest")
if not version.startswith("SageMath version 10.9"):
    raise SystemExit(f"unexpected SageMath version: {version}")

runtime={"schema_version":"sagemath-runtime-v1","image":digest,"source_tag":"sagemath/sagemath:10.9","sage_version":version,"network_policy":"none","read_only_root":True,"cap_drop_all":True,"no_new_privileges":True,"pids_limit":128,"memory_mb":3072,"cpus":2,"timeout_seconds":90}
(C/"sagemath-runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(C/"requirements-sagemath.txt").write_text("# SageMath runs from the repository-pinned official container image.\n",encoding="utf-8")

module=r'''#!/usr/bin/env python3
"""Bounded SageMath operations executed in an exact-digest offline container."""
from __future__ import annotations
import json,math,os,re,subprocess,tempfile
from collections.abc import Mapping,Sequence
from pathlib import Path
from typing import Any
from compute_runner import ComputeError
HERE=Path(__file__).resolve().parent
RUNTIME=json.loads((HERE/"sagemath-runtime.json").read_text(encoding="utf-8"))
MODES={"simplify","solve","differentiate","integrate","matrix_analysis","number_theory"}
ALLOWED_NAMES={"sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","exp","log","sqrt","abs","pi","e"}
EXPRESSION_RE=re.compile(r"^[A-Za-z0-9+*/^()., \t-]{1,2000}$")
NAME_RE=re.compile(r"[A-Za-z][A-Za-z0-9]*")
VAR_RE=re.compile(r"^[A-Za-z][A-Za-z0-9]{0,15}$")

def mapping(v:Any,n:str)->Mapping[str,Any]:
 if not isinstance(v,Mapping):raise ComputeError(f"{n} must be an object")
 return v

def sequence(v:Any,n:str,max_items:int)->Sequence[Any]:
 if isinstance(v,(str,bytes)) or not isinstance(v,Sequence) or not 1<=len(v)<=max_items:raise ComputeError(f"{n} must contain 1 to {max_items} items")
 return v

def integer(v:Any,n:str,lo:int,hi:int)->int:
 if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi:raise ComputeError(f"{n} must be an integer between {lo} and {hi}")
 return v

def finite(v:Any,n:str)->float:
 if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):raise ComputeError(f"{n} must be finite")
 return float(v)

def variables(raw:Any)->list[str]:
 rows=[] if raw in (None,[]) else list(sequence(raw,"inputs.variables",20))
 result=[]
 for i,v in enumerate(rows):
  name=str(v)
  if not VAR_RE.fullmatch(name) or name in ALLOWED_NAMES or name in result:raise ComputeError(f"invalid or duplicate variable at inputs.variables[{i}]")
  result.append(name)
 return result

def expression(v:Any,vars_:list[str])->str:
 s=str(v or "").strip()
 if not EXPRESSION_RE.fullmatch(s) or "**" in s:raise ComputeError("expression contains forbidden characters or syntax")
 names=set(NAME_RE.findall(s));unknown=sorted(names-set(vars_)-ALLOWED_NAMES)
 if unknown:raise ComputeError(f"expression contains non-allowlisted names: {unknown}")
 return s

def payload(inputs:Mapping[str,Any])->dict[str,Any]:
 mode=str(inputs.get("mode") or "")
 if mode not in MODES:raise ComputeError(f"inputs.mode must be one of {sorted(MODES)}")
 p={"mode":mode}
 if mode in {"simplify","solve","differentiate","integrate"}:
  vars_=variables(inputs.get("variables"));p["variables"]=vars_;p["expression"]=expression(inputs.get("expression"),vars_)
  var=str(inputs.get("variable") or (vars_[0] if len(vars_)==1 else ""))
  if mode!="simplify" and var not in vars_:raise ComputeError("inputs.variable must be one of inputs.variables")
  if var:p["variable"]=var
  if mode=="differentiate":p["order"]=integer(inputs.get("order",1),"inputs.order",1,10)
  if mode=="integrate":
   has_low="lower" in inputs;has_high="upper" in inputs
   if has_low!=has_high:raise ComputeError("lower and upper must be supplied together")
   if has_low:p["lower"]=finite(inputs["lower"],"inputs.lower");p["upper"]=finite(inputs["upper"],"inputs.upper")
 elif mode=="matrix_analysis":
  rows=sequence(inputs.get("matrix"),"inputs.matrix",20);matrix=[];width=None
  for i,row in enumerate(rows):
   parsed=[finite(x,f"inputs.matrix[{i}]") for x in sequence(row,f"inputs.matrix[{i}]",20)]
   width=width or len(parsed)
   if len(parsed)!=width:raise ComputeError("matrix rows must have equal length")
   matrix.append(parsed)
  p["matrix"]=matrix
 elif mode=="number_theory":
  action=str(inputs.get("action") or "")
  if action not in {"factor","is_prime","gcd","lcm","euler_phi"}:raise ComputeError("unsupported number_theory action")
  values=list(sequence(inputs.get("values"),"inputs.values",20));parsed=[]
  for i,v in enumerate(values):parsed.append(integer(v,f"inputs.values[{i}]",-10**18,10**18))
  if action in {"factor","is_prime","euler_phi"} and len(parsed)!=1:raise ComputeError(f"{action} requires exactly one value")
  p.update({"action":action,"values":parsed})
 return p

RUNNER=r'''from sage.all import *
import json
p=json.load(open('/work/payload.json'))
mode=p['mode']
loc={'sin':sin,'cos':cos,'tan':tan,'asin':asin,'acos':acos,'atan':atan,'sinh':sinh,'cosh':cosh,'tanh':tanh,'exp':exp,'log':log,'sqrt':sqrt,'abs':abs,'pi':pi,'e':e}
for name in p.get('variables',[]):loc[name]=var(name)
def expr():return sage_eval(p['expression'].replace('^','**'),locals=loc)
if mode=='simplify':out={'expression':str(expr().full_simplify())}
elif mode=='solve':out={'solutions':[str(x) for x in solve(expr()==0,loc[p['variable']])]}
elif mode=='differentiate':out={'derivative':str(diff(expr(),loc[p['variable']],p['order'])),'order':p['order']}
elif mode=='integrate':
 x=loc[p['variable']]
 value=integral(expr(),x,p['lower'],p['upper']) if 'lower' in p else integral(expr(),x)
 out={'integral':str(value),'definite':'lower' in p}
elif mode=='matrix_analysis':
 A=matrix(SR,p['matrix']);ev=A.eigenvalues() if A.nrows()==A.ncols() else []
 out={'rows':A.nrows(),'columns':A.ncols(),'rank':A.rank(),'determinant':str(A.det()) if A.nrows()==A.ncols() else None,'characteristic_polynomial':str(A.charpoly()) if A.nrows()==A.ncols() else None,'eigenvalues':[str(x) for x in ev]}
elif mode=='number_theory':
 vals=[Integer(x) for x in p['values']];a=p['action']
 if a=='factor':v=str(factor(vals[0]))
 elif a=='is_prime':v=bool(vals[0].is_prime())
 elif a=='gcd':v=int(gcd(vals))
 elif a=='lcm':v=int(lcm(vals))
 else:v=int(euler_phi(vals[0]))
 out={'action':a,'value':v}
print(json.dumps({'engine':'SageMath','mode':mode,'result':out},ensure_ascii=False))
'''

def run_sage(p:Mapping[str,Any])->dict[str,Any]:
 if os.environ.get("SAGEMATH_FIXTURE_MODE")=="1":return {"engine":"SageMath-fixture","mode":p["mode"],"result":{"validated":True}}
 image=str(RUNTIME.get("image") or "")
 if not re.fullmatch(r"sagemath/sagemath@sha256:[0-9a-f]{64}",image):raise ComputeError("repository-pinned SageMath image is invalid")
 with tempfile.TemporaryDirectory(prefix="compute-sagemath-") as tmp:
  root=Path(tmp);(root/"payload.json").write_text(json.dumps(dict(p),ensure_ascii=False),encoding="utf-8");(root/"runner.py").write_text(RUNNER,encoding="utf-8")
  for f in root.iterdir():f.chmod(0o644)
  cmd=["docker","run","--rm","--network","none","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges","--pids-limit",str(RUNTIME["pids_limit"]),"--memory",f"{RUNTIME['memory_mb']}m","--cpus",str(RUNTIME["cpus"]),"--tmpfs","/tmp:rw,noexec,nosuid,size=256m","-v",f"{root}:/work:ro","--entrypoint","sage",image,"-python","/work/runner.py"]
  try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=int(RUNTIME["timeout_seconds"]),check=False)
  except (OSError,subprocess.TimeoutExpired) as exc:raise ComputeError(f"SageMath runtime failed: {exc}") from exc
  if r.returncode!=0:raise ComputeError(f"SageMath container returned {r.returncode}: {r.stderr[-1500:]}")
  try:out=json.loads(r.stdout.strip().splitlines()[-1])
  except Exception as exc:raise ComputeError("SageMath returned invalid JSON") from exc
  out["runtime"]={"image":image,"sage_version":RUNTIME["sage_version"],"network_policy":"none","arbitrary_code_allowed":False}
  return out

def symbolic_mathematics(inputs:Mapping[str,Any])->dict[str,Any]:return run_sage(payload(mapping(inputs,"inputs")))
OPERATIONS={"symbolic_mathematics":symbolic_mathematics}
'''
(C/"sagemath_operations.py").write_text(module,encoding="utf-8")

test=r'''import json,os,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import sagemath_operations as sage
class SageMathTests(unittest.TestCase):
 def setUp(self):os.environ["SAGEMATH_FIXTURE_MODE"]="1"
 def tearDown(self):os.environ.pop("SAGEMATH_FIXTURE_MODE",None)
 def test_all_modes_are_bounded(self):
  samples=[{"mode":"simplify","variables":["x"],"expression":"(x^2-1)/(x-1)"},{"mode":"solve","variables":["x"],"variable":"x","expression":"x^2-4"},{"mode":"differentiate","variables":["x"],"variable":"x","expression":"sin(x)*exp(x)","order":2},{"mode":"integrate","variables":["x"],"variable":"x","expression":"x^2","lower":0,"upper":1},{"mode":"matrix_analysis","matrix":[[1,2],[3,4]]},{"mode":"number_theory","action":"factor","values":[360]}]
  for sample in samples:self.assertEqual(sage.symbolic_mathematics(sample)["mode"],sample["mode"])
 def test_expression_injection_is_rejected(self):
  for bad in ["__import__(x)","x;system(x)","x.__class__","[x]"]:
   with self.assertRaises(Exception):sage.symbolic_mathematics({"mode":"simplify","variables":["x"],"expression":bad})
 def test_exact_digest_and_offline_policy(self):
  r=json.loads((HERE/"sagemath-runtime.json").read_text());self.assertRegex(r["image"],r"^sagemath/sagemath@sha256:[0-9a-f]{64}$");self.assertEqual(r["network_policy"],"none")
if __name__=="__main__":unittest.main()
'''
(C/"tests/test_sagemath_operations.py").write_text(test,encoding="utf-8")

registry=json.loads((C/"tool-registry.json").read_text(encoding="utf-8"))
registry["groups"]=[g for g in registry["groups"] if g.get("id")!="sagemath-symbolic"]
modes={
 "simplify":{"maturity":"controlled-preview","network_policy":"deny","deterministic":True,"limits":{"max_expression_characters":2000,"max_variables":20}},
 "solve":{"maturity":"controlled-preview","network_policy":"deny","deterministic":True,"limits":{"max_expression_characters":2000,"max_variables":20}},
 "differentiate":{"maturity":"controlled-preview","network_policy":"deny","deterministic":True,"limits":{"max_expression_characters":2000,"max_order":10}},
 "integrate":{"maturity":"controlled-preview","network_policy":"deny","deterministic":True,"limits":{"max_expression_characters":2000}},
 "matrix_analysis":{"maturity":"controlled-preview","network_policy":"deny","deterministic":True,"limits":{"max_rows":20,"max_columns":20}},
 "number_theory":{"maturity":"controlled-preview","network_policy":"deny","deterministic":True,"limits":{"max_values":20,"max_absolute_integer":10**18}}
}
registry["groups"].append({"id":"sagemath-symbolic","module":"sagemath_operations","operations":["symbolic_mathematics"],"input_validation":"mode_allowlist","default_requirements":["requirements-sagemath.txt"],"mode_requirements":{},"network_policy":"deny","deterministic":True,"maturity":"controlled-preview","resource_limits":{"max_seconds":90,"max_memory_mb":3072},"rollback":{"stable_module":"sagemath_operations","strategy":"git-revert"},"modes":modes})
(C/"tool-registry.json").write_text(json.dumps(registry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

caps=json.loads((C/"compute-capabilities.json").read_text(encoding="utf-8"))
old_modes=int(caps["managed_mode_count"]);old_eff=int(caps["effective_managed_mode_count"])
caps["runtime_packages"]["SageMath"]="10.9 official container image pinned by exact repository digest; Docker network=none, read-only root and no-new-privileges"
caps["operations"]=[o for o in caps["operations"] if o.get("id")!="symbolic_mathematics"]
caps["operations"].append({"id":"symbolic_mathematics","engine":"SageMath 10.9 exact-digest offline container","availability":"controlled-preview","use_when":"exact symbolic simplification, equations, calculus, matrix algebra or number theory are required","typical_output":"exact symbolic expressions, solutions, derivatives, integrals, matrix invariants or number-theory results"})
caps["operation_count"]=len(caps["operations"]);caps["managed_mode_count"]=old_modes+6 if "symbolic mathematics" not in " ".join(caps.get("toolkit_assessment",{}).get("scope",[])).lower() else old_modes;caps["effective_managed_mode_count"]=old_eff+6 if caps["managed_mode_count"]==old_modes+6 else old_eff
scope=caps["toolkit_assessment"]["scope"]
line="exact symbolic algebra, equation solving, calculus, matrix invariants and number theory in a pinned offline SageMath runtime"
if line not in scope:scope.append(line)
(C/"compute-capabilities.json").write_text(json.dumps(caps,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

matrix=json.loads((C/"systems-computation-matrix.json").read_text(encoding="utf-8"))
matrix["routes"]["symbolic_mathematics"]={"problem_class":"exact symbolic mathematics","system_level":"mechanism","feedback_structure":"closed-form symbolic transformation with exact-runtime verification","required_gates":["input_quality","assumption_register","constraint_feasibility","external_validation"]}
(C/"systems-computation-matrix.json").write_text(json.dumps(matrix,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

readme=C/"sagemath-guide.md"
readme.write_text(f"""# SageMath受控符号计算能力\n\n- 版本：{version}\n- 镜像：`{digest}`\n- 网络：Docker `--network none`，同时由计算中心外层网络命名空间断网。\n- 文件系统：只读根文件系统；仅只读挂载固定运行脚本和结构化输入。\n- 安全：禁用任意Python/Sage代码，只接受受限表达式字符、变量白名单和固定6种模式。\n- 资源：90秒、3072MB、2 CPU、128 PID。\n- 模型/API调用：0。\n""",encoding="utf-8")
print(json.dumps({"status":"PASS","image":digest,"sage_version":version,"operation":"symbolic_mathematics","modes":sorted(modes)},ensure_ascii=False))
