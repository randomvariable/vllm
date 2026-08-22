# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReSET under speculative decoding: chain-scan equivalence with the
sequential policy.

`resolve_reset_speculative` resolves a per-position temperature across each
request's draft chain and defers the running-state commit until acceptance is
known. The contract: for every acceptance outcome, the temperatures applied
to committed positions and the committed state are exactly what sequential
ReSET (`resolve_reset`, one call per committed token) produces over the
accepted draft prefix plus the recovered token. This makes speculative
decoding a pure speedup for ReSET requests rather than a different policy.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.reset import (
    _RUNNING_FIELDS,
    make_reset_state,
    resolve_reset,
    resolve_reset_speculative,
)

VOCAB = 48
NL_ID = 3
DNL_ID = 5


def _luts(device):
    nl = torch.zeros(VOCAB, dtype=torch.bool, device=device)
    dnl = torch.zeros(VOCAB, dtype=torch.bool, device=device)
    nl[NL_ID] = True
    dnl[DNL_ID] = True
    return nl, dnl


class _HostBackedTensor:
    """CPU stand-in for UvaBackedTensor (UVA requires an NVIDIA driver)."""

    def __init__(self, size, dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        return self.gpu[:n] if n is not None else self.gpu


@pytest.fixture
def host_states(monkeypatch):
    from vllm.v1.worker.gpu.sample import states as states_mod

    monkeypatch.setattr(states_mod, "UvaBackedTensor", _HostBackedTensor)
    return states_mod


def _single_row_state(device, t_low=0.1, t_high=1.0, tau0=0.6, window=4):
    state = make_reset_state(1, device)
    state.enabled[0] = 1
    state.t_low[0] = t_low
    state.t_high[0] = t_high
    state.tau0[0] = tau0
    state.window[0] = window
    return state


def _sequential_oracle(
    logits_rows, committed, last_committed, gen0, nl, dnl, device, **cfg
):
    """Sequential ReSET over committed tokens: one resolve_reset call each."""
    state = _single_row_state(device, **cfg)
    temps = []
    last = last_committed
    for j, tok in enumerate(committed):
        last_t = torch.tensor([last], dtype=torch.int64, device=device)
        step = torch.tensor([gen0 + j], dtype=torch.int64, device=device)
        temps.append(
            resolve_reset(logits_rows[j : j + 1], last_t, step, nl, dnl, state).clone()
        )
        last = int(tok)
    return torch.cat(temps), state


def _assert_state_matches(scan_snap, oracle_state, row=0):
    for name in _RUNNING_FIELDS:
        got = scan_snap[name][row]
        want = getattr(oracle_state, name)[0]
        if got.is_floating_point():
            torch.testing.assert_close(got, want, msg=lambda m: f"{name}: {m}")
        else:
            assert torch.equal(got, want), f"{name}: {got} vs {want}"


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    chain=st.integers(min_value=1, max_value=6),
    gen0=st.sampled_from([0, 1, 3, 4, 7]),
)
@settings(max_examples=40, deadline=None)
def test_scan_matches_sequential_oracle_all_acceptance_points(seed, chain, gen0):
    """Every acceptance point: committed rows and state match the oracle."""
    device = torch.device("cpu")
    nl, dnl = _luts(device)
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(chain, VOCAB, generator=g) * 2.0
    drafts = torch.randint(0, VOCAB, (max(chain - 1, 1),), generator=g)
    last_committed = int(torch.randint(0, VOCAB, (1,), generator=g))

    scan_logits = logits.clone()
    state = _single_row_state(device)
    snapshots = resolve_reset_speculative(
        scan_logits,
        base_rows=torch.zeros(1, dtype=torch.int64),
        chain_lens=torch.tensor([chain]),
        last_committed=torch.tensor([last_committed]),
        draft_ids=drafts.unsqueeze(0),
        base_gen_step=torch.tensor([gen0]),
        nl_lut=nl,
        dnl_lut=dnl,
        state=state,
        max_chain=chain,
    )
    assert len(snapshots) == chain

    for r in range(chain):
        # r accepted drafts plus one recovered token at the stop position.
        recovered = int(torch.randint(0, VOCAB, (1,), generator=g))
        committed = drafts[:r].tolist() + [recovered]
        oracle_temps, oracle_state = _sequential_oracle(
            logits, committed, last_committed, gen0, nl, dnl, device
        )
        for j in range(r + 1):
            torch.testing.assert_close(
                scan_logits[j], logits[j] / oracle_temps[j], msg=lambda m: f"{m}"
            )
        _assert_state_matches(snapshots[r], oracle_state)


def test_recovered_token_drives_next_step_boundary():
    """The recovered token (not the rejected draft) is next step's last token.

    Two chained engine steps: the recovered token at the stop point is a
    newline while the rejected draft token is not, so the next step's boundary
    detection only matches the oracle if it sees the recovered token.
    """
    device = torch.device("cpu")
    nl, dnl = _luts(device)
    g = torch.Generator().manual_seed(7)
    chain = 3
    logits1 = torch.randn(chain, VOCAB, generator=g)
    logits2 = torch.randn(chain, VOCAB, generator=g)
    # Draft chain: [plain, NL, plain]; walk stops at position 1 (the NL draft
    # is rejected) and the recovered token there is also NL -- the oracle sees
    # [plain, NL(recovered)], so step 2 starts after two... one NL in a row.
    drafts1 = torch.tensor([10, NL_ID, 12])
    recovered = NL_ID
    last_committed = 9
    gen0 = 5

    state = _single_row_state(device)
    scan_logits1 = logits1.clone()
    snaps1 = resolve_reset_speculative(
        scan_logits1,
        base_rows=torch.zeros(1, dtype=torch.int64),
        chain_lens=torch.tensor([chain]),
        last_committed=torch.tensor([last_committed]),
        draft_ids=drafts1.unsqueeze(0),
        base_gen_step=torch.tensor([gen0]),
        nl_lut=nl,
        dnl_lut=dnl,
        state=state,
        max_chain=chain,
    )
    committed1 = [10, recovered]
    oracle_t1, oracle_state = _sequential_oracle(
        logits1, committed1, last_committed, gen0, nl, dnl, device
    )
    _assert_state_matches(snaps1[1], oracle_state)

    # Step 2: the engine presents the recovered token as last_committed.
    committed_state = make_reset_state(1, device)
    committed_state.enabled[0] = 1
    committed_state.window[0] = 4
    committed_state.tau0[0] = 0.6
    for name in _RUNNING_FIELDS:
        getattr(committed_state, name)[0] = snaps1[1][name][0]
    drafts2 = torch.tensor([20, 21, 22])
    gen1 = gen0 + len(committed1)
    scan_logits2 = logits2.clone()
    snaps2 = resolve_reset_speculative(
        scan_logits2,
        base_rows=torch.zeros(1, dtype=torch.int64),
        chain_lens=torch.tensor([chain]),
        last_committed=torch.tensor([committed1[-1]]),
        draft_ids=drafts2.unsqueeze(0),
        base_gen_step=torch.tensor([gen1]),
        nl_lut=nl,
        dnl_lut=dnl,
        state=committed_state,
        max_chain=chain,
    )
    # Oracle continues from its own carried state: re-run step 1 then step 2.
    _, oracle_state2 = _sequential_oracle(
        logits1, committed1, last_committed, gen0, nl, dnl, device
    )
    committed2 = drafts2.tolist()
    oracle_temps2 = []
    last = committed1[-1]
    for j, tok in enumerate(committed2):
        last_t = torch.tensor([last], dtype=torch.int64)
        step = torch.tensor([gen1 + j], dtype=torch.int64)
        oracle_temps2.append(
            resolve_reset(logits2[j : j + 1], last_t, step, nl, dnl, oracle_state2)
        )
        last = tok
    for j in range(chain):
        torch.testing.assert_close(
            scan_logits2[j], logits2[j] / oracle_temps2[j], msg=lambda m: f"{m}"
        )
    _assert_state_matches(snaps2[chain - 1], oracle_state2)


def test_double_newline_boundary_mid_chain():
    """A dnl draft token resets the within-step buffer mid-chain.

    Position 1's resolve must observe the boundary on draft token 0, matching
    the oracle's sequential reset exactly.
    """
    device = torch.device("cpu")
    nl, dnl = _luts(device)
    g = torch.Generator().manual_seed(17)
    chain = 4
    logits = torch.randn(chain, VOCAB, generator=g) * 2.0
    drafts = torch.tensor([DNL_ID, 8, 9])
    gen0 = 3

    scan_logits = logits.clone()
    state = _single_row_state(device)
    snapshots = resolve_reset_speculative(
        scan_logits,
        base_rows=torch.zeros(1, dtype=torch.int64),
        chain_lens=torch.tensor([chain]),
        last_committed=torch.tensor([2]),
        draft_ids=drafts.unsqueeze(0),
        base_gen_step=torch.tensor([gen0]),
        nl_lut=nl,
        dnl_lut=dnl,
        state=state,
        max_chain=chain,
    )
    # Full acceptance: the whole draft chain plus... the oracle walks the
    # same committed tokens (all drafts, recovered at the last position).
    committed = drafts[: chain - 1].tolist() + [14]
    oracle_temps, oracle_state = _sequential_oracle(
        logits, committed, 2, gen0, nl, dnl, device
    )
    # The dnl boundary must have fired for the oracle at position 1.
    assert oracle_state.step_len[0] == chain - 1
    for j in range(chain):
        torch.testing.assert_close(
            scan_logits[j], logits[j] / oracle_temps[j], msg=lambda m: f"{m}"
        )
    _assert_state_matches(snapshots[chain - 1], oracle_state)


def test_ragged_multi_request_chains_leave_other_rows_untouched():
    """Ragged chains: per-request commits match; non-ReSET rows are exact."""
    device = torch.device("cpu")
    nl, dnl = _luts(device)
    g = torch.Generator().manual_seed(11)
    lens = [1, 3, 2]
    num_rows = sum(lens) + 2  # plus a non-ReSET request's two rows
    logits = torch.randn(num_rows, VOCAB, generator=g)
    cu = torch.tensor([0, 1, 4, 6, 8], dtype=torch.int32)  # 4th req: rows 6..7
    drafts = torch.randint(0, VOCAB, (3, 2), generator=g)
    last = torch.tensor([1, 2, 3], dtype=torch.int64)
    gen0 = torch.tensor([0, 2, 9], dtype=torch.int64)

    scan_logits = logits.clone()
    state = make_reset_state(3, device)
    state.enabled[:] = 1
    state.window[:] = 4
    snapshots = resolve_reset_speculative(
        scan_logits,
        base_rows=cu[:-1][:3].to(torch.int64),
        chain_lens=(cu[1:] - cu[:-1]).to(torch.int64)[:3],
        last_committed=last,
        draft_ids=drafts,
        base_gen_step=gen0,
        nl_lut=nl,
        dnl_lut=dnl,
        state=state,
        max_chain=max(lens),
    )
    torch.testing.assert_close(scan_logits[6:], logits[6:])

    gen = torch.Generator().manual_seed(13)
    for req, chain in enumerate(lens):
        r = int(torch.randint(0, chain, (1,), generator=gen))
        recovered = int(torch.randint(0, VOCAB, (1,), generator=gen))
        committed = drafts[req, :r].tolist() + [recovered]
        rows = logits[cu[req] : cu[req] + chain]
        oracle_temps, oracle_state = _sequential_oracle(
            rows, committed, int(last[req]), int(gen0[req]), nl, dnl, device
        )
        for j in range(r + 1):
            torch.testing.assert_close(
                scan_logits[cu[req] + j],
                rows[j] / oracle_temps[j],
                msg=lambda m: f"{m}",
            )
        _assert_state_matches(snapshots[r], oracle_state, row=req)


def _fake_req_states(device, num_reqs, prompt_len, total_len, last_tok, drafts):
    return SimpleNamespace(
        prompt_len=SimpleNamespace(gpu=prompt_len.to(device)),
        total_len=SimpleNamespace(gpu=total_len.to(device)),
        last_sampled_tokens=last_tok.to(device),
        draft_tokens=drafts.to(device),
    )


def test_states_staging_apply_and_commit(host_states):
    """SamplingStates wires staging, the spec scan, and the deferred commit."""
    device = torch.device("cpu")
    torch.manual_seed(23)
    states = host_states.SamplingStates(3, VOCAB, seed=0)
    prompt_len = torch.tensor([10, 10, 10], dtype=torch.int32)
    total_len = torch.tensor([12, 15, 10], dtype=torch.int32)
    last_tok = torch.zeros(3, 1, dtype=torch.int64)
    drafts = torch.tensor([[30, 31], [32, 33], [34, 35]], dtype=torch.int64)
    states.bind_reasoning_state(
        _fake_req_states(device, 3, prompt_len, total_len, last_tok, drafts),
        None,
        None,
        device,
        model_config=None,
    )
    nl, dnl = _luts(device)
    states._reset_nl_lut = nl
    states._reset_dnl_lut = dnl

    # Request 0: ReSET. Request 1: plain. Request 2: ReSET.
    states.add_request(
        0, SamplingParams(temperature=1.0, temperature_low=0.2, reset_window=4)
    )
    states.add_request(1, SamplingParams(temperature=0.7))
    states.add_request(
        2, SamplingParams(temperature=1.0, temperature_low=0.2, reset_window=4)
    )
    assert states.use_reset[0] and states.use_reset[2]
    assert not states.use_reset[1]
    assert states.reset_state.t_low[0].item() == pytest.approx(0.2)

    # Requests 0 and 2 decode with 3-row chains; request 1 is single-row.
    cu = torch.tensor([0, 3, 4, 7], dtype=torch.int32)
    logits = torch.randn(7, VOCAB)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    idx_mapping_np = idx_mapping.numpy()
    scan_logits = logits.clone()
    states.apply_reset(scan_logits, idx_mapping, idx_mapping_np, cu, max_chain=3)

    # Commit: request 0 commits 2 tokens, request 2 commits all 3.
    states.commit_reset_spec(torch.tensor([2, 1, 3], dtype=torch.int32))

    for req, n_committed in ((0, 2), (2, 3)):
        base = int(cu[req])
        committed = drafts[req, : n_committed - 1].tolist() + [40]
        oracle_temps, oracle_state = _sequential_oracle(
            logits[base : base + 3],
            committed,
            0,
            int(total_len[req] - prompt_len[req]),
            nl,
            dnl,
            device,
            t_low=0.2,
            window=4,
        )
        for j in range(n_committed):
            torch.testing.assert_close(
                scan_logits[base + j],
                logits[base + j] / oracle_temps[j],
                msg=lambda m: f"{m}",
            )
        for name in _RUNNING_FIELDS:
            got = getattr(states.reset_state, name)[req]
            want = getattr(oracle_state, name)[0]
            if got.is_floating_point():
                torch.testing.assert_close(got, want, msg=lambda m: f"{name}: {m}")
            else:
                assert torch.equal(got, want), f"{name}: {got} vs {want}"
    # Request 1 (plain) row is untouched by the scan; its static 0.7 divide
    # happens later in apply_temperature, which is not under test here.
    torch.testing.assert_close(scan_logits[3], logits[3])


def test_commit_with_zero_sampled_keeps_prior_state(host_states):
    """A request that committed nothing keeps its pre-step state."""
    device = torch.device("cpu")
    torch.manual_seed(29)
    states = host_states.SamplingStates(1, VOCAB, seed=0)
    states.bind_reasoning_state(
        _fake_req_states(
            device,
            1,
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 2, dtype=torch.int64),
        ),
        None,
        None,
        device,
        model_config=None,
    )
    nl, dnl = _luts(device)
    states._reset_nl_lut = nl
    states._reset_dnl_lut = dnl
    states.add_request(0, SamplingParams(temperature=1.0, temperature_low=0.2))
    prior = {
        name: getattr(states.reset_state, name).clone() for name in _RUNNING_FIELDS
    }

    logits = torch.randn(2, VOCAB)
    states.apply_reset(
        logits.clone(),
        torch.tensor([0], dtype=torch.int32),
        np.array([0]),
        torch.tensor([0, 2], dtype=torch.int32),
        max_chain=2,
    )
    states.commit_reset_spec(torch.tensor([0], dtype=torch.int32))
    for name in _RUNNING_FIELDS:
        torch.testing.assert_close(
            getattr(states.reset_state, name), prior[name], msg=lambda m: f"{m}"
        )
