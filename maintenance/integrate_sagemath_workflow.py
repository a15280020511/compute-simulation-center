#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
workflow = ROOT / ".github/workflows/compute-ticket.yml"
text = workflow.read_text(encoding="utf-8")

cache_line = "            compute-center/requirements-sagemath.txt\n"
if cache_line not in text:
    marker = "            compute-center/requirements-sumo.txt\n"
    if marker not in text:
        raise SystemExit("compute requirements cache insertion marker not found")
    text = text.replace(marker, cache_line + marker, 1)

step = '''      - name: Pre-pull and attest selected SageMath runtime
        if: steps.prepare.outputs.accepted == 'true'
        shell: bash
        run: |
          set -euo pipefail
          if ! jq -e '.capability_pack == "sagemath-symbolic"' compute-artifacts/compute-runtime-plan.json >/dev/null; then
            exit 0
          fi
          image="$(jq -r '.image' compute-center/sagemath-runtime.json)"
          expected_prefix="$(jq -r '.sage_version_expected_prefix' compute-center/sagemath-runtime.json)"
          test "$image" != "null"
          docker pull "$image"
          observed_digest="$(docker image inspect "$image" --format '{{index .RepoDigests 0}}')"
          test "$observed_digest" = "$image"
          observed_version="$(docker run --rm --network none --entrypoint sage "$image" --version | head -n 1)"
          [[ "$observed_version" == "$expected_prefix"* ]]
          jq -n \
            --arg image "$image" \
            --arg observed_digest "$observed_digest" \
            --arg observed_version "$observed_version" \
            '{schema_version:"sagemath-runtime-attestation-v1",image:$image,observed_digest:$observed_digest,observed_version:$observed_version,network_policy:"none",model_calls:0,external_data_fetches:0}' \
            > compute-artifacts/sagemath-runtime-attestation.json

'''
anchor = "      - name: Comment compute ticket accepted\n"
if "Pre-pull and attest selected SageMath runtime" not in text:
    if anchor not in text:
        raise SystemExit("SageMath pre-pull insertion marker not found")
    text = text.replace(anchor, step + anchor, 1)

workflow.write_text(text, encoding="utf-8")
print("integrated SageMath production workflow")
