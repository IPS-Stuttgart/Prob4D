provider = provider.replace(
    '"schema": "prob4d.dot-r04-r10-checkpoint-compatibility"',
    '"schema": "prob4d.dot-r11-r30-checkpoint-compatibility"',
    1,
)
provider = provider.replace(
    'dot-rope-cut3r-heldout-runtime-gpuserver6000-',
    'dot-rope-query-selective-runtime-gpuserver6000-',
    1,
)

dataset_step = r'''      - name: Resolve, download, and verify official R11-R30 archives
        id: dataset
        shell: bash
        run: |
          set -euo pipefail
          root="${{ steps.workspace.outputs.root }}"
          cache="$RUNTIME_CACHE_ROOT/dot-v29"
          /usr/bin/mkdir -p "$cache"
          ARCHIVE_TABLE="$root/tmp/archives.tsv" \
            RECEIPT="$root/evidence/official-archives.json" \
            python - <<'PY'
          import json
          import os
          import urllib.request
          from pathlib import Path

          expected = {
              os.environ["R11_ARCHIVE"]: os.environ["R11_MD5"],
              os.environ["R21_ARCHIVE"]: os.environ["R21_MD5"],
          }
          request = urllib.request.Request(
              os.environ["DATASET_API"],
              headers={"User-Agent": "Prob4D-DOT-R11-R30-provider/1"},
          )
          with urllib.request.urlopen(request, timeout=60) as response:
              metadata = json.load(response)
          if metadata.get("status") != "OK":
              raise SystemExit("Dataverse metadata request did not return OK")
          files = metadata["data"]["latestVersion"]["files"]
          by_name = {entry["dataFile"]["filename"]: entry["dataFile"] for entry in files}
          rows = []
          receipt = {}
          for name, md5 in expected.items():
              data_file = by_name.get(name)
              if data_file is None:
                  raise SystemExit(f"official DOT metadata lacks {name}")
              checksum = data_file.get("checksum") or {}
              if checksum.get("type") != "MD5" or checksum.get("value", "").lower() != md5:
                  raise SystemExit(f"official checksum changed for {name}")
              item = {
                  "datafile_id": int(data_file["id"]),
                  "byte_count": int(data_file["filesize"]),
                  "checksum": checksum,
              }
              receipt[name] = item
              rows.append((name, item["datafile_id"], item["byte_count"], md5))
          Path(os.environ["RECEIPT"]).write_text(
              json.dumps(receipt, indent=2, sort_keys=True) + "\n",
              encoding="utf-8",
          )
          Path(os.environ["ARCHIVE_TABLE"]).write_text(
              "".join(f"{name}\t{file_id}\t{size}\t{md5}\n" for name, file_id, size, md5 in rows),
              encoding="utf-8",
          )
          PY

          while IFS=$'\t' read -r name file_id bytes md5; do
            archive="$cache/$name"
            valid=false
            if [[ -f "$archive" && "$(stat -c %s "$archive")" = "$bytes" ]]; then
              measured=$(md5sum "$archive" | awk '{print $1}')
              [[ "$measured" = "$md5" ]] && valid=true
            fi
            if [[ "$valid" != true ]]; then
              part="$archive.partial"
              if [[ -f "$part" ]]; then
                part_bytes=$(stat -c %s "$part")
                if (( part_bytes > bytes )); then
                  /usr/bin/rm -f -- "$part"
                fi
              fi
              curl --fail --location --retry 8 --retry-all-errors \