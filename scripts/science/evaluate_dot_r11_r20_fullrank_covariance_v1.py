#!/usr/bin/env python3
"""Score fixed-mean correlated CUT3R uncertainty on DOT R11-R20."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, traceback, zipfile
from pathlib import Path
from typing import Any
import numpy as np
from prob4d.dependence_tempering import temper_shared_dependence
from prob4d.dot_rope_cut3r_study import content_id

REV="c64765ea766e667a566e1b565e8ed01ffd734e53"
REQ="75e9cac8d776dac0575dc29d922b1468542cf3bf0ad9949810fc18441341863d"
SEAL="38ea78e8bf44cbeaedeeadaee862af3cc6369d35d7e3b5a2b5fac0f020c7145b"
RANK="a1fc018dc7fb504b35f6fbfc422e7a59edaeb71009c6f29c512019d42f949ced"
CAMERAS={"cam001":["R13","R14","R15","R16","R18","R19"],"cam002":["R17"],"cam005":["R11","R12","R20"]}
BUNDLES={"cam001":"14d78d100a62e11e0e51d76491e72f9bb34d30e141a8a510d9bdb69bacb35dff","cam002":"e0c476a039e9610a2849c6dba3996936fbf7f648b68b61d93bc0519fff389dae","cam005":"60c56f8c1cc2b07e418d67b01d81c3dc7dd7b369dbeaac9899f029af8f79b818"}
EVAL={"overlap_frames":[3,4,5],"metric_fit_a_frames":[1,2,3],"metric_fit_b_frames":[5,6,7],"score_frames":[6,7]}
UNC={"bootstrap_replicates":256,"bootstrap_seed":20260830,"means_held_fixed":True,"observation_noise_fraction":0.02,"orbit_nodes":33,"probe_count":8,"probe_radius_fraction_of_provider_span":0.25,"scalar_inflation":4.0,"tensor_gh_order":3}

def load(path:Path,name:str)->Any:
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None: raise RuntimeError(f"cannot load {path}")
 m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def read(path:Path)->dict[str,Any]:
 v=json.loads(path.read_text())
 if type(v) is not dict: raise ValueError(path)
 return v

def write(path:Path,v:dict[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()

def member(seq:str,dim:int,frame:int,camera:str)->str:
 cam=camera if dim==2 else "cam001"
 return f"{seq}/coordinates/{dim}d/frame{frame:06d}_{cam}.txt"

def aggregate(rows:list[dict[str,Any]], allowed:set[str])->list[dict[str,Any]]:
 out=[]
 for method in sorted({r["method"] for r in rows}):
  x=[r for r in rows if r["method"]==method and r["sequence"] in allowed]
  if not x: continue
  out.append({"method":method,"sequence_count":len(x),"mean_nll_per_dimension":float(np.mean([r["normalized_nll_per_dimension"] for r in x])),"coverage_95":float(np.mean([r["covered_95"] for r in x])),"mean_mahalanobis":float(np.mean([r["mahalanobis"] for r in x])),"mean_sd_fraction_of_span":float(np.mean([r["mean_predictive_sd_fraction_of_span"] for r in x]))})
 return sorted(out,key=lambda r:r["mean_nll_per_dimension"])

def compare(rows:list[dict[str,Any]], allowed:set[str], comparator:str, seed:int)->dict[str,Any]:
 a={r["sequence"]:r["normalized_nll_per_dimension"] for r in rows if r["method"]=="dependence_alpha_0850" and r["sequence"] in allowed}
 b={r["sequence"]:r["normalized_nll_per_dimension"] for r in rows if r["method"]==comparator and r["sequence"] in allowed}
 names=sorted(set(a)&set(b)); d=np.array([a[n]-b[n] for n in names],float)
 rng=np.random.default_rng(seed); boot=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1)
 return {"comparator":comparator,"sequences":names,"mean_difference":float(d.mean()),"lower_95":float(np.quantile(boot,.025)),"upper_95":float(np.quantile(boot,.975)),"wins":int((d<0).sum()),"per_sequence":{n:float(a[n]-b[n]) for n in names}}

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--provider-root",type=Path,required=True); p.add_argument("--rank-result",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--repository-revision",required=True); a=p.parse_args()
 if a.repository_revision!=REV: raise ValueError("revision changed")
 root=a.dataset_root.resolve(strict=True); archive_path=(root/"R11-20.zip").resolve(strict=True); archive_path.relative_to(root)
 providers=a.provider_root.resolve(strict=True); rank=read(a.rank_result.resolve(strict=True))
 if rank["result_id"]!=RANK: raise ValueError("rank result changed")
 rank_by={r["sequence"]:r for r in rank["per_sequence"]}
 seal=read(providers/"providers/provider-seal.json")
 if seal["provider_seal_id"]!=SEAL: raise ValueError("provider seal changed")
 scripts=Path(__file__).resolve().parent; base=load(scripts/"run_dot_rope_cut3r_native_provider.py","fullrank_base"); pooled=load(scripts/"evaluate_dot_rope_cut3r_pooled.py","fullrank_pooled")
 pooled._ACTIVE_COORDINATE_COLUMNS=(0,1); pooled._ACTIVE_COORDINATE_MODE="pixel-zero-based"; pooled.SUPPORT_RULE={"overlap_minimum_total_common":8,"overlap_minimum_nonempty_frames":2,"provider_truth_minimum_total":6,"score_minimum_total":2}
 base._ORIGINAL_LOAD_RUN=base._load_run; base._load_run=lambda bundle,record: pooled._load_run_with_metadata(base,bundle,record); base.parse_coordinate_text=pooled._parse_coordinate_text; base._sample_markers=pooled._sample_markers; base._collect_pair=pooled._collect_pair; base._collect_provider_truth=pooled._collect_provider_truth
 original=base.covariance_closures
 def closures(*x,**kw):
  v=original(*x,**kw); v["dependence_alpha_0850"]=temper_shared_dependence(v["pointwise_quadratic"],v["shared_quadratic_curvature"],.85); return v
 base.covariance_closures=closures
 sequence_results=[]; method_rows=[]; failures=[]; provider_ids={}
 with zipfile.ZipFile(archive_path) as z:
  names=set(z.namelist())
  for camera,sequences in CAMERAS.items():
   protocol=read(providers/f"components/protocols/component-{camera}.json"); bundle=providers/f"providers/{camera}/bundle"; manifest=base._verify_provider_bundle(bundle,protocol)
   if manifest["provider_bundle_id"]!=BUNDLES[camera] or manifest["request_id"]!=REQ or manifest["prob4d_revision"]!=REV: raise ValueError(f"{camera} identity changed")
   provider_ids[camera]=manifest["provider_bundle_id"]; records={s:{} for s in sequences}
   for r in manifest["outputs"]: records[r["sequence"]][r["run"]]=r
   for seq in sequences:
    frames={}
    for f in range(1,8):
     m2,m3=member(seq,2,f,camera),member(seq,3,f,camera)
     if m2 not in names or m3 not in names: raise ValueError(f"missing markers {seq}/{f}")
     frames[f]=(pooled._parse_coordinate_text(z.read(m2).decode(),2),pooled._parse_coordinate_text(z.read(m3).decode(),3))
    runs={n:base._load_run(bundle,records[seq][n]) for n in ("continuous","window_a","window_b")}; eval_protocol=dict(protocol); eval_protocol["evaluation"]=EVAL; eval_protocol["uncertainty"]=UNC
    pooled._MARKER_DIAGNOSTICS.clear(); pooled._COLLECTION_DIAGNOSTICS.clear()
    try: result, rows=base._sequence_evaluation(seq,runs,frames,eval_protocol)
    except Exception as e:
     failures.append({"sequence":seq,"camera":camera,"error":f"{type(e).__name__}: {e}","traceback_tail":traceback.format_exc().splitlines()[-8:]}); continue
    rr=rank_by[seq]; result.update({"camera":camera,"factor_rank":rr["factor_rank"],"rank_six_supported":rr["supported"],"observable_condition_ratio":rr["observable_condition_ratio"]}); sequence_results.append(result)
    method_rows += [{**r,"camera":camera,"factor_rank":rr["factor_rank"]} for r in rows]
 all_seq={r["sequence"] for r in sequence_results}; full={s for s in all_seq if rank_by[s]["factor_rank"]==7}; comparators=["pointwise_quadratic","shared_quadratic_curvature","local_first_order","cluster_bootstrap_fallback"]
 result={"schema":"prob4d.dot-r11-r20-fullrank-correlated-covariance-diagnostic","schema_version":1,"decision":"complete-source-diagnostic" if len(all_seq)==10 else "partial-source-diagnostic","repository_revision":a.repository_revision,"request_id":REQ,"provider_seal_id":SEAL,"provider_bundle_ids":provider_ids,"rank_result_id":RANK,"archive":{"name":"R11-20.zip","md5":"23ce3e7067465d3edabe20b4c7cfa388","sha256":sha(archive_path)},"evaluation_profile":EVAL,"selected_method":"dependence_alpha_0850","selected_alpha":.85,"evaluated_sequences":sorted(all_seq),"full_rank_sequences":sorted(full),"aggregate_all":aggregate(method_rows,all_seq),"aggregate_full_rank_only":aggregate(method_rows,full),"comparisons_all":{c:compare(method_rows,all_seq,c,20260902+i) for i,c in enumerate(comparators)},"comparisons_full_rank_only":{c:compare(method_rows,full,c,20261902+i) for i,c in enumerate(comparators)},"sequence_results":sequence_results,"method_rows":method_rows,"failures":failures,"information_boundary":{"r11_r20_provider_predictions_reused":True,"r11_r20_source_markers_opened":True,"r21_r70_payloads_opened":False,"means_held_fixed_across_methods":True,"rank_six_required_for_scoring":False,"bayesian_phystwin_executed":False,"causal4d_executed":False},"claim_boundary":"Post-source diagnostic; not a fresh held-out confirmation and does not override the terminal rank-six source gate."}
 result["diagnostic_id"]=content_id(result); a.output_dir.mkdir(parents=True,exist_ok=False); write(a.output_dir/"result.json",result)
 lines=["# DOT R11–R20 full-rank correlated covariance","",f"Decision: **{result['decision']}**",f"Diagnostic ID: `{result['diagnostic_id']}`","","| Method | N | Mean NLL/dim | Coverage | Mean SD/span |","|---|---:|---:|---:|---:|"]
 for r in result["aggregate_all"]: lines.append(f"| {r['method']} | {r['sequence_count']} | {r['mean_nll_per_dimension']:.6f} | {r['coverage_95']:.3f} | {r['mean_sd_fraction_of_span']:.6f} |")
 lines += ["","## Dependence-alpha comparisons on rank-7 sequences",""]
 for c,v in result["comparisons_full_rank_only"].items(): lines.append(f"- vs `{c}`: {v['mean_difference']:.6f} [{v['lower_95']:.6f}, {v['upper_95']:.6f}], wins {v['wins']}/{len(v['sequences'])}")
 if failures: lines += ["","## Failures",""]+[f"- {v['sequence']}: {v['error']}" for v in failures]
 lines += ["",result["claim_boundary"],""]; (a.output_dir/"SUMMARY.md").write_text("\n".join(lines))
 print(json.dumps({"decision":result["decision"],"diagnostic_id":result["diagnostic_id"],"evaluated_sequences":len(all_seq),"full_rank_sequences":len(full),"best_method":result["aggregate_all"][0]["method"] if result["aggregate_all"] else None,"full_rank_vs_pointwise":result["comparisons_full_rank_only"].get("pointwise_quadratic"),"failures":failures},indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
