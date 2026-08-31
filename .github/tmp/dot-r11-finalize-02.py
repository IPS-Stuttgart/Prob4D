                --connect-timeout 30 --continue-at - \
                --output "$part" "$DATAFILE_API_ROOT/$file_id"
              test "$(stat -c %s "$part")" = "$bytes"
              printf '%s  %s\n' "$md5" "$part" | md5sum --check --strict
              /usr/bin/mv -- "$part" "$archive"
            fi
            test "$(stat -c %s "$archive")" = "$bytes"
            printf '%s  %s\n' "$md5" "$archive" | md5sum --check --strict
          done < "$root/tmp/archives.tsv"
          echo "root=$cache" >> "$GITHUB_OUTPUT"

'''
provider = replace_section(
    provider,
    '      - name: Download and verify the official DOT R01-R10 archive\n',
    '      - name: Build and attest native CUT3R RoPE in an isolated checkout\n',
    dataset_step,
)

predict_and_upload = r'''      - name: Predict R11-R30 from normal-view images only
        id: execute
        continue-on-error: true
        shell: bash
        env:
          RUNTIME_PYTHON: ${{ steps.runtime.outputs.python }}
          CHECKPOINT_PATH: ${{ steps.checkpoint.outputs.path }}
          DATASET_ROOT: ${{ steps.dataset.outputs.root }}
          CUT3R_CHECKOUT: ${{ steps.rope.outputs.checkout }}
          REQUEST_ID: ${{ needs.authorize.outputs.request_id }}
          EXPECTED_HEAD_SHA: ${{ needs.authorize.outputs.head_sha }}
        run: |
          set -euo pipefail
          root="${{ steps.workspace.outputs.root }}"
          PYTHONPATH="$GITHUB_WORKSPACE/src:$GITHUB_WORKSPACE/scripts/science:$CUT3R_CHECKOUT:$CUT3R_CHECKOUT/src" \
            "$RUNTIME_PYTHON" \
            scripts/science/run_dot_rope_query_selective_heldout.py \
              predict \
              --protocol "$PROTOCOL_PATH" \
              --request-id "$REQUEST_ID" \
              --prob4d-revision "$EXPECTED_HEAD_SHA" \
              --dataset-root "$DATASET_ROOT" \
              --cut3r-checkout "$CUT3R_CHECKOUT" \
              --checkpoint "$CHECKPOINT_PATH" \
              --runtime-receipt "$root/evidence/runtime-receipt.json" \
              --output-dir "$root/provider/bundle"

      - name: Read sealed provider result
        id: result
        if: always()
        shell: bash
        env:
          EXECUTION_OUTCOME: ${{ steps.execute.outcome }}
        run: |
          set -euo pipefail
          root="${{ steps.workspace.outputs.root }}"
          manifest="$root/provider/bundle/manifest.json"
          if [[ "$EXECUTION_OUTCOME" != "success" || ! -f "$manifest" ]]; then
            ROOT="$root/provider" OUTCOME="${EXECUTION_OUTCOME:-not-run}" python - <<'PY'
          import json
          import os
          from pathlib import Path

          root = Path(os.environ["ROOT"])
          root.mkdir(parents=True, exist_ok=True)
          (root / "technical-failure.json").write_text(
              json.dumps(
                  {
                      "decision": "technical-failure",
                      "execution_outcome": os.environ["OUTCOME"],
                      "marker_payloads_opened": False,
                  },
                  indent=2,
                  sort_keys=True,
              )
              + "\n",
              encoding="utf-8",
          )
          PY
            echo "decision=technical-failure" >> "$GITHUB_OUTPUT"
            echo "provider_bundle_id=unavailable" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          MANIFEST="$manifest" python - <<'PY'
          import json
          import os
