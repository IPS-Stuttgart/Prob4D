from pathlib import Path

ROOT = Path('.')
query_path = ROOT / '.github/workflows/dot-rope-query-selective-heldout-v1.yml'
r04_path = ROOT / '.github/workflows/dot-rope-cut3r-heldout-confirmation-v1.yml'
test_path = ROOT / 'tests/test_dot_rope_query_selective_heldout_workflow.py'

def section(text: str, start: str, end: str) -> tuple[int, int, str]:
    a = text.index(start)
    b = text.index(end, a)
    return a, b, text[a:b]

def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    a, b, _ = section(text, start, end)
    return text[:a] + replacement + text[b:]

query = query_path.read_text(encoding='utf-8')
r04 = r04_path.read_text(encoding='utf-8')

old_env = '''  DATASET_ROOT: /mnt/seagate10tb/florianpfaff/datasets/dot
  R11_ARCHIVE: R11-20.zip
  R11_MD5: 23ce3e7067465d3edabe20b4c7cfa388
  R21_ARCHIVE: R21-30.zip
  R21_MD5: 8aee77f79d1aff6e1f3fd21886b251a0
  DATASET_API: https://dataverse.orc.gmu.edu/api/datasets/:persistentId/?persistentId=doi:10.13021/ORC2020/XXLVXM
  DATAFILE_API_ROOT: https://dataverse.orc.gmu.edu/api/access/datafile
  CUT3R_RUNTIME_CHECKOUT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/CUT3R
  CUT3R_RUNTIME_PYTHON: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/venv/bin/python
  CUT3R_RUNTIME_CHECKPOINT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/cut3r_512_dpt_4_64.pth
  CUT3R_RUNTIME_MANIFEST: /home/github-runner/.cache/prob4d/cut3r-runtime-v1/bootstrap-manifest.json
  CUT3R_REVISION: 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
  CUT3R_CHECKPOINT_SHA256: 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103
'''
new_env = '''  R11_ARCHIVE: R11-20.zip
  R11_MD5: 23ce3e7067465d3edabe20b4c7cfa388
  R21_ARCHIVE: R21-30.zip
  R21_MD5: 8aee77f79d1aff6e1f3fd21886b251a0
  DATASET_API: https://dataverse.orc.gmu.edu/api/datasets/:persistentId/?persistentId=doi:10.13021/ORC2020/XXLVXM
  DATAFILE_API_ROOT: https://dataverse.orc.gmu.edu/api/access/datafile
  CUT3R_REVISION: 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
  CUT3R_CHECKPOINT_SHA256: 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103
  CUT3R_CHECKPOINT_GDRIVE_ID: 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD
  RUNTIME_CACHE_ROOT: /home/github-runner/.cache/prob4d/dot-r11-r30-cut3r-gpuserver6000-v1
  RETAINED_CUT3R_CHECKOUT: ${{ vars.CUT3R_CHECKOUT }}
  RETAINED_CUT3R_CHECKPOINT: ${{ vars.CUT3R_CHECKPOINT }}
  RETAINED_CUT3R_PYTHON: ${{ vars.CUT3R_PYTHON }}
  PIP_DISABLE_PIP_VERSION_CHECK: "1"
'''
if query.count(old_env) != 1:
    raise SystemExit('query environment preimage changed')
query = query.replace(old_env, new_env, 1)

_, _, provider = section(r04, '\n  provider:\n', '\n  evaluate:\n')
provider = provider[1:]
provider = provider.replace(
    '    name: Seal marker-free R04-R10 CUT3R predictions on gpuserver6000',
    '    name: Seal marker-free R11-R30 CUT3R point maps on gpuserver6000',
    1,
)
provider = provider.replace('    needs: authorize', '    needs: [authorize, prerequisite]', 1)
provider = provider.replace('    timeout-minutes: 480', '    timeout-minutes: 720', 1)
provider = provider.replace(
    '''    outputs:
      provider_bundle_id: ${{ steps.seal.outputs.provider_bundle_id }}
      provider_artifact_name: ${{ steps.seal.outputs.provider_artifact_name }}
''',
    '''    outputs:
      decision: ${{ steps.result.outputs.decision }}
      provider_bundle_id: ${{ steps.result.outputs.provider_bundle_id }}
''',
    1,
)
provider = provider.replace('prob4d-dot-r04-r10-', 'prob4d-dot-r11-r30-')