          import re
          from pathlib import Path

          value = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
          if value.get("decision") != "sealed-r11-r30-provider-predictions":
              raise SystemExit("R11-R30 provider did not seal predictions")
          bundle_id = value.get("provider_bundle_id", "")
          if re.fullmatch(r"[0-9a-f]{64}", bundle_id) is None:
              raise SystemExit("R11-R30 provider bundle identity is invalid")
          boundary = value.get("information_boundary") or {}
          if boundary.get("two_dimensional_markers_opened") is not False:
              raise SystemExit("provider opened 2-D markers")
          if boundary.get("three_dimensional_markers_opened") is not False:
              raise SystemExit("provider opened 3-D markers")
          with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
              output.write("decision=sealed-r11-r30-provider-predictions\n")
              output.write(f"provider_bundle_id={bundle_id}\n")
          PY
          /usr/bin/chmod -R a-w "$root/provider"

      - name: Upload immutable R11-R30 provider bundle
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7
        with:
          name: dot-rope-query-selective-provider-${{ github.run_id }}-${{ github.run_attempt }}
          path: ${{ steps.workspace.outputs.root }}/provider/
          if-no-files-found: error
          retention-days: 30

'''
provider = replace_section(
    provider,
    '      - name: Predict from marker-free normal-view images only\n',
    '      - name: Upload bounded runtime receipts\n',
    predict_and_upload,
)
provider = provider.replace(
    '      - name: Upload bounded runtime receipts',
    '      - name: Upload bounded gpuserver6000 runtime receipts',
    1,
)
provider = provider.replace(
    '      - name: Remove isolated prediction workspace',
    '''      - name: Require sealed provider predictions
        if: always()
        shell: bash
        env:
          DECISION: ${{ steps.result.outputs.decision }}
        run: |
          set -euo pipefail
          test "$DECISION" = "sealed-r11-r30-provider-predictions"

      - name: Remove isolated provider workspace''',
    1,
)
provider = provider.replace(
    'expected="${RUNNER_TEMP}/prob4d-dot-r11-r30-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"',
    'expected="${RUNNER_TEMP}/prob4d-dot-r11-r30-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"',
)

query = replace_section(query, '\n  provider:\n', '\n  seal:\n', '\n' + provider)
query_path.write_text(query, encoding='utf-8')

test = test_path.read_text(encoding='utf-8')
test = test.replace(
    'selector = "runs-on: [self-hosted, Linux, X64, gpuserver4090]"',
    'selector = "runs-on: [self-hosted, gpuserver6000]"',
)
test = test.replace(
    'assert \'test "$RUNNER_NAME" = "workstation1"\' in provider',
    'assert \'test "$RUNNER_NAME" = "workstation2"\' in provider',
)
test = test.replace(
    '    assert "Predict R11-R30 from normal-view images only" in provider\n',
    '    assert "Resolve, download, and verify official R11-R30 archives" in provider\n'
    '    assert "--continue-at -" in provider\n'
    '    assert "Build and attest native CUT3R RoPE in an isolated checkout" in provider\n'
    '    assert "Predict R11-R30 from normal-view images only" in provider\n',
)
test_path.write_text(test, encoding='utf-8')
