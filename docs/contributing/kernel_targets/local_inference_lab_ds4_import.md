# DSV4-Flash vLLM 0.27 reconciliation

## Pinned inputs

| Input | Value |
| --- | --- |
| Rebased upstream base | `upstream/main` `d6941300fcb9d4a8bbea19f8b610c2aff9fc5cc3` |
| Pre-rebase history | `backup/homelabs-main-pre-027-rebase-20260812` `b50183e8b2cfdffd22376b888dc6e7d12d45325d` |
| Fork squash commit | `e6d6c9236a66a73cde7e9080a0f651f4fb1aa6de` |
| Reconciliation hotfix commit | `b66a77ced894bdeb059ad6df3e91b1c455a2a749` |
| B12X submodule | `9bbae67841e4818e7472e1edcdca8ebcbda68611` |
| DeepGEMM submodule | `1a8a73ce75c8c4de0eed46dab0ca25f92509b51a` |
| FlashInfer submodule | `d9594fad157a31180096cf664e51c01010b21a52` |
| Mia source | `c9a84e173ad39f4d7a0e2632f7d2d44290b68306` |

At this documentation revision, the fork branch is four commits atop
`upstream/main`: the squash, reconciliation hotfix, lineage correction, and
this final lineage record. The backup retains the prior history. Mia's source
is read with `git show` at the pinned commit, not the stale checked-out recipe
revision.

## Fix reconciliation

| Fix | Mia source | Fork owner | Disposition |
| --- | --- | --- | --- |
| #49486 short-context skip-topk | `patches/hotfix-dsv4-skip-topk-49486.sh` | `DeepseekV4Indexer.forward` in `vllm/models/deepseek_v4/attention.py` | Ported, with a DCP single-rank guard. |
| #50312 MTP hidden buffer | `patches/hotfix-dsv4-mtp-buffer-50312.sh` | `DeepseekV4Model` and V1 GPU model runner | Ported; allocation and copies are None-safe when no speculator consumes them. |
| #50004 adaptive C128A topk width | `patches/hotfix-dsv4-adaptive-topk-50004.sh` | `DeepseekV4FlashMLAMetadataBuilder` in `vllm/models/deepseek_v4/sparse_mla.py` | Ported. |
| #48957 empty C128 compressor skip | `patches/hotfix-dsv4-skip-empty-c128-48957.sh` | `DeepseekCompressor` in `vllm/models/deepseek_v4/compressor.py` | Ported; disabled for full CUDA graphs. |
| #50298 FlashMLA workspace reuse | no new port required | `vllm/models/deepseek_v4/nvidia/flashmla.py` and `common/ops/cache_utils.py` | Retained from fork delta. |
| #48407 dense-prefill indexer skip | `patches/hotfix-dsv4-dense-prefill-indexer-48407.sh` | no dense-MHA binding exists | Dormant; not ported or activated. |
| Issue #22 `nvfp4_ds_mla` decode | recipe Issue #22 | `mla_attention.py` canonicalization | Skipped as moot: `nvfp4_ds_mla` becomes `fp8_ds_mla` before the implementation. |
| Issue #21 `encode_arguments_to_dsml` | recipe Issue #21 | Hugging Face checkpoint encoding | Out of scope. |

## C128A width and CUDA graphs

#50004 keeps its large preallocated buffers for stable addresses but passes
packed views and a per-batch active stride to the C128A metadata kernel. The
width derives from `CommonAttentionBatchTopology.max_seq_len_upper_bound` when
available, otherwise `CommonAttentionMetadata.max_seq_len`. Full CUDA graph
capture uses the same upper-bound value to construct its batch topology, so a
captured shape and every replay of that shape derive the same packed width and
stride. It does not read `get_forward_context()` during metadata construction:
metadata is built before that context is installed.

## Intentional non-ports

#48407 remains dormant. This fork has no dense-MHA route or
`dense_mha_metadata_layer_name` binding in `DeepseekV4Indexer`; its prerequisite
that top-k output is unconsumed is therefore not established. Do not add a
binding until a dense-MHA route is implemented and verified.

Issue #22 remains deferred because the native FlashMLA reader has no separate
NVFP4 path: `nvfp4_ds_mla` is canonicalized to `fp8_ds_mla` before the
implementation. Do not broaden the existing cache-type equality until that
reader exists.

## Local verification and remaining qualification

`python3 scripts/verify-dsv4-027-equality-gate.py` passes with zero failures.
It is CPU-only and proves source shape and boundary arithmetic: the #49486 DCP
guard, short-context boundary, all-candidate index-set arithmetic, and #48407
dormancy. It does not execute a ported Triton kernel.

The two kernel pytest targets cannot run in the current local environment:
`PYTHONPATH=. .venv/bin/python -m pytest -q
tests/kernels/test_compressor_kv_cache.py` fails at collection, and
`PYTHONPATH=. .venv/bin/python -m pytest -q
tests/kernels/attention/test_flashmla_sparse.py` has three import failures.
Both stop at `nixl_ep.Buffer`, which is absent from the installed empty
`nixl_ep` namespace. The environment is also behind
`requirements/cuda.txt` for Humming (`humming-kernels==0.1.10` installed;
`==0.1.12` required); after a temporary `Buffer` shim, imports stop at missing
`humming.dtypes`. These are environment failures, not source-test results.

GPU decode, throughput, accuracy, and full CUDA graph replay qualification
must run on the DGX Spark pair or CUDA CI, not this workstation.
