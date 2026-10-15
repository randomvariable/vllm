# Shared-Memory Message Queue

`MessageQueue` in `vllm/distributed/device_communicators/shm_broadcast.py` is
vLLM's one-writer, many-reader channel between processes. It carries the
engine's per-step RPCs to its workers and the workers' replies back, and it
backs `GroupCoordinator.broadcast_object` for the TP, DCP and inner-DP world
groups. This page describes what uses it, how a message moves, the wake-up
protocol, the conditions under which a reader can miss a wake-up, the formal
models that check the protocol, and its measured cost.

## Producers and consumers

| Queue | Writer | Readers | Traffic |
| --- | --- | --- | --- |
| `rpc_broadcast_mq` (`MultiprocExecutor`, `RayExecutorV2`) | EngineCore executor | every worker of the DP replica | one message per `collective_rpc`, so one or more per scheduler step (`execute_model`, `sample_tokens`) |
| `worker_response_mq` (one per worker) | that worker | the executor | one reply per RPC from each replying rank; `execute_model` replies only from the output rank |
| `create_single_reader_mq_broadcasters` (inner DP world, multi-node) | every worker | the output-rank worker | cross-node replies relayed to the executor |
| `GroupCoordinator.mq_broadcaster` (`tp`, `dcp`, `inner_dp_world`) | group rank 0 | the other group ranks | `broadcast_object` and the metadata half of `broadcast_tensor_dict`; used at start-up, in warm-up, and for infrequent syncs such as adaptive-verification cost curves |

`check_shm_free_space` from the same module also guards the shared-memory
allocations of the CPU KV offload region and the DeepSeek Engram tables.

The hot path is the executor loop: the executor enqueues an RPC, every worker
dequeues it, runs the step, and the output rank enqueues its reply, which the
executor dequeues. With async scheduling, up to `max_concurrent_batches` RPCs
(2, or the PP size plus 1 with model runner V2) are outstanding at once.

## Data path

Local readers share an `ShmRingBuffer` of `max_chunks` slots of
`max_chunk_bytes` each. Each slot has a metadata row: one written flag and one
read flag per local reader. The writer waits until every reader has read a
slot, clears the written flag, copies the payload, clears the read flags, and
sets the written flag. A reader waits for written-and-unread, copies the
payload, and sets its read flag. Memory fences order the flag and payload
stores. After the metadata rows, the buffer holds one parked flag and one
fence-capability flag per local reader, used by the wake-up protocol below.

Two zmq XPUB data sockets carry what does not fit the ring:

- **Overflow.** A payload larger than a slot marks the slot as overflow and
  follows on `local_socket`. The reader releases the slot, then waits for the
  payload on the socket.
- **Remote readers.** Readers on other nodes receive every message on the
  `remote_socket` TCP XPUB. Nothing acknowledges these messages.

Both data sockets set `XPUB_NODROP`. A default XPUB silently drops a message
when a reader's pipe looks full, and neither a released overflow slot nor a
remote reader would ever see that message again. With `XPUB_NODROP`, a
blocking send waits for room instead, and the high water mark still bounds the
queued memory.

## Waiting

A reader that finds its slot empty spins for its grace period, then parks in
`zmq.Poller.poll` on its notification socket. The grace is adaptive: an EMA
of the interval between reads sets it between
`VLLM_SHM_BROADCAST_ADAPTIVE_MIN_GRACE_MS` (0.05 ms) and
`VLLM_SHM_BROADCAST_ADAPTIVE_MAX_GRACE_MS` (2 ms); see
[Inter-process communication spin tuning](../configuration/env_vars.md#inter-process-communication-spin-tuning).
An explicit `busy_loop_s`, including 0, pins it. The writer spins within its
own adaptive grace while readers hold the next slot, then polls the flags in
sleeps that double up to `VLLM_SHM_BROADCAST_WRITE_PARK_MAX_MS`.

After a write, the writer sends a one-byte ping on its notification socket.
Each reader's SUB socket sets `CONFLATE`, so pings that arrive while the reader
spins collapse into one, and a parked reader consumes the single pending ping
when it wakes. Every park is capped at `SHM_READER_RECHECK_INTERVAL_MS` (5 s),
after which the reader re-reads the flags. A writer waiting for readers to
free a slot polls the flags, so it never depends on a notification.

The executor waits on several reply queues at once in `_gather_responses`. It
probes each with `ready()`, which reads the flags and consumes nothing, and
parks on the first pending queue in 50 ms slices with `wait_for_message`.
A single reply queue uses a plain `dequeue`.

`shutdown()` sets `shutting_down` and sends the reader's cancel ping on an
inproc PAIR socket. Every reader wait polls that socket: the ring park, the
overflow-payload wait on `local_socket`, and a remote reader's wait on
`remote_socket`. The reader then raises `RuntimeError("cancelled")`.

### Wake-up invariant

A reader re-reads the flags after every wake-up and before every park, so the
protocol is correct as long as one property holds:

> A parked reader whose slot is ready always has a notification pending: in
> its `CONFLATE` slot, in flight, or about to be sent by the writer.

If a notification is lost, the parked reader sleeps until the recheck: up to
5 s in `dequeue`, or one 50 ms slice in `_gather_responses`. In the executor
loop that stalls the whole step: the executor waits in `get_response` while
every worker waits in `dequeue`.

### How zmq can lose a notification

A zmq PUB socket drops a message, without an error, in two cases:

1. **Stale high water mark view.** The PUB drops a message when its view of
   the pipe says the pipe is full. The view is refreshed only when the socket
   processes its commands. A send does that at most once every ~3M CPU-counter
   ticks (~1 ms on x86, ~3 ms on aarch64), and the pipe reader returns credit
   only every `(hwm + 1) / 2` drained messages. With `SNDHWM=1`, a ping sent
   within that window of the previous one is dropped even after the reader has
   consumed the previous one. Any finite high water mark can drop a ping in the
   same way after enough sends without a refresh.
2. **Unknown subscription.** The PUB drops a message for a subscriber whose
   subscription it has not yet processed. The start-up handshake in
   `wait_until_ready` used to cover only `local_socket`, so the first pings
   could race the readers' notification subscriptions.

The original fixed 1 s grace hid both: a reader rarely parked within
milliseconds of a read or of start-up. A grace of at most 2 ms parks within
that window routinely, so both fixes below are required by the adaptive
default.

#### History

The notification socket came with upstream
[PR #28053](https://github.com/vllm-project/vllm/pull/28053), which removed
the idle busy loop for shared IBM Z hosts. It set `SNDHWM=1` so that "PUB
sockets will silently drop subsequent messages" while a reader is busy:
dropping was meant as rate limiting, on the assumption that a full pipe means
the reader already has an unread ping. The stale view above breaks that
assumption. [PR #58917](https://github.com/vllm-project/vllm/pull/58917)
reverts to the default high water mark. The parked-flag protocol below
recovers the saving that `SNDHWM=1` was after, without dropping pings.

### Notification channel

The writer's notification socket is an `XPUB` with `SNDHWM=0` (no limit) and
`XPUB_VERBOSE`, and `wait_until_ready` waits for every local reader's
notification subscription before it sends `READY`:

- `SNDHWM=0` removes the drop condition for the first case. The queue stays
  bounded: `CONFLATE` collapses pings on the reader side, the writer's I/O
  thread drains the pipe, and a reader that stops reading stops releasing ring
  slots, which blocks the writer after `max_chunks` writes.
- The subscription handshake removes the second case. `XPUB_VERBOSE` is
  required because every reader sends the same empty subscription.

### Parked-flag protocol

A ping is useful only to a reader that is parked. When every local reader can
issue a full fence, the writer pings only if some reader's parked flag is set:

| Step | Reader (`MessageQueue._park`) | Writer (`MessageQueue._notify_readers`) |
| --- | --- | --- |
| 1 | store `parked = 1` | store `written = 1` (in `acquire_write`) |
| 2 | full fence | full fence |
| 3 | re-check the slot; if ready, clear the flag and read | load the parked flags |
| 4 | otherwise poll, then clear the flag | ping if any flag is set |

This is a Dekker handshake: with both fences, either the writer sees the
parked flag and pings, or the reader's re-check sees the written flag. Two
details carry the correctness:

- **Every poll is flagged.** `_park` reads the spin grace once and then either
  yields or runs the four steps. A first implementation also let
  `SpinCondition.wait` re-read the grace; when the grace expired between the
  two reads, the reader polled without its flag, the writer skipped the ping,
  and the reader slept 5 s. The benchmark below found it; `message_queue.qnt`
  now models it as `skipIdleFlaglessPark`.
- **Step 2 is a store-to-load fence.** Each step 3 load must not be satisfied
  before the other process's step 1 store is visible. `memory_fence()`, which
  takes a `threading.Lock`, gives acquire and release ordering but not that
  guarantee on weakly ordered CPUs. The `vllm.spinloop` extension therefore
  provides `full_fence()`, a C11 `__atomic_thread_fence(__ATOMIC_SEQ_CST)`:

| Architecture | Instruction |
| --- | --- |
| x86-64 | `lock or` (or `mfence`) |
| AArch64 | `dmb ish` |
| POWER (ppc64le) | `sync` |
| IBM Z (s390x) | `bcr 15,0` |
| RISC-V | `fence rw,rw` |

### Futex wake

On Linux the extension also exports futex park and wake calls on 4-byte
words in the ring buffer, one per local reader. The handshake is the same,
with the parked flag replaced by the futex word and the notify ping by a
futex wake:

- `park_prepare` stores 1 in the reader's word and fences. The reader then
  re-checks its slot and `shutting_down`, and calls `park_wait`, which sleeps
  in `FUTEX_WAIT` only while the word is still 1, for at most the recheck
  interval.
- `wake_parked` fences, then exchanges every set word to 0 and calls
  `FUTEX_WAKE` on it.

The kernel compares the word under its futex lock, so a wake that lands
between the reader's re-check and its sleep makes the sleep return at once.
A parked reader wakes in one futex syscall instead of a zmq send, an I/O
thread hand-off, and a `poll` return: on the benchmark host the one-way
parked latency drops from 61 to 34 µs at p50 and from 1.4 ms to 0.18 ms at
p99. `shutdown()` wakes a futex-parked reader through its own word.

#### Shallow parking

Most of a parked reader's wake latency at a 10 ms step gap is not the futex
but the CPU leaving a deep idle state: on the benchmark host C3 exits in
350 µs, against 18 µs for C2. The idle governor picks the state from the
time until the next timer, and avoids a state whose target residency
(700 µs for C3) exceeds it. So while the next write is likely, `park_wait`
sleeps in futex slices of `VLLM_SHM_BROADCAST_SHALLOW_SLICE_MS` (0.5 ms),
looping in C without returning to Python. The window lasts until twice the
observed interval between reads after the last read, at most
`VLLM_SHM_BROADCAST_SHALLOW_MAX_MS` (50 ms); after it the reader sleeps
until woken, so idle CPU stays at zero. Each slice costs one timer wake-up of
a few microseconds. At a 10 ms gap this cut the round trip from 443 to 54 µs
with one worker and from 331 to 98 µs with four, with no rise in worker CPU.
It needs no privileges. Set the slice below the deepest state's target
residency on other CPUs (`/sys/devices/system/cpu/cpu*/cpuidle/state*/residency`).

The futex words are node-local, like the ring. Readers on other nodes keep
receiving every message on `remote_socket`, so cross-node collectives are
unaffected.

### Wake mode selection

Each reader records its capability before it subscribes: 0 without the
extension, 1 with `full_fence`, 2 with futexes as well. In
`wait_until_ready` the writer picks the wake mode that every local reader
supports, publishes it in the ring buffer, and only then sends `READY`, after
which each reader reads it:

| Mode | Condition | Wake-up |
| --- | --- | --- |
| futex | writer and every local reader have futexes (Linux) | `wake_parked` |
| parked-only ping | writer and every local reader have `full_fence` | notify ping while a parked flag is set |
| ping | otherwise, for example XPU builds or Python 3.10 | notify ping after every write |

The notify socket and its subscription handshake stay in place in every mode.

### Ring back-pressure

The executor cannot read replies while it waits for a free RPC slot, and a
worker cannot read RPCs while it waits for a free reply slot. The loop
therefore deadlocks if more RPCs are outstanding than the rings can hold:
roughly more than `max_chunks` of `rpc_broadcast_mq` plus `max_chunks` of the
reply queue plus the one being processed. With 10 slots per ring and at most
`max_concurrent_batches` outstanding RPCs, vLLM is far from that bound.

## Performance

`benchmarks/benchmark_shm_broadcast.py` drives the executor loop with public
`MessageQueue` APIs: one executor process broadcasts a 1 KiB RPC to N worker
processes, each worker replies on its own queue, and the executor waits for
every reply before the next step, after an idle gap of 0, 1 or 10 ms. An idle
scenario connects the queues and sends nothing for 10 s. Every process is
pinned to its own physical core. CPU is user plus system time from
`/proc/<pid>/stat`, including zmq's I/O threads, divided by the wall time of
the window.

The comparison covers three variants, each with its own defaults:

- `main`: upstream `main` at `429a3da665`: fixed 1 s reader grace, PUB with
  `SNDHWM=1`.
- `arm-no`: this tree without the `vllm.spinloop` extension, so the writer
  pings after every write.
- `arm-so`: this tree with the extension, so parked-only pings are active.

Host: AMD Ryzen AI Max+ 395 (Zen 5, 16 cores, SMT 2), Linux 7.2.9-cachyos,
`amd-pstate-epp` with the `performance` governor, Python 3.12.13, pyzmq 27.2.0,
libzmq 4.3.5. Each figure is the median of 3 repeats, interleaved by variant.

### CPU

Worker CPU is summed over the workers, in cores (CPU-seconds per second).

| Scenario | Workers | `main` | `arm-no` | `arm-so` |
| --- | --- | --- | --- | --- |
| idle, 10 s | 1 | 0.099 | 0.000 | 0.000 |
| idle, 10 s | 4 | 0.392 | 0.000 | 0.000 |
| 10 ms gap | 1 | 0.996 | 0.023 | 0.026 |
| 10 ms gap | 4 | 3.985 | 0.085 | 0.087 |
| 1 ms gap | 1 | 0.998 | 0.832 | 0.827 |
| 1 ms gap | 4 | 3.990 | 3.124 | 3.197 |
| back-to-back | 4 | 3.981 | 3.975 | 3.984 |

Upstream readers spin for 1 s after every read, so a reader burns a full core
whenever steps arrive less than 1 s apart, and 0.1 core while idle. The
adaptive grace parks a reader within 2 ms, which cuts worker CPU about 46x at
a 10 ms step gap and removes idle CPU. At a 1 ms gap the grace still covers
most of the interval, so the saving is about 20%. Back-to-back traffic keeps
every reader busy in all variants. The executor uses about 1 core
back-to-back and under 0.04 core when paced, in all variants.

### Latency and throughput

| Scenario | Workers | Metric | `main` | `arm-no` | `arm-so` |
| --- | --- | --- | --- | --- | --- |
| back-to-back | 1 | steps/s | 74 784 | 57 919 | 66 999 |
| back-to-back | 1 | round trip p50 / p99 | 13 / 16 µs | 16 / 22 µs | 14 / 18 µs |
| back-to-back | 4 | steps/s | 37 741 | 21 413 | 34 466 |
| back-to-back | 4 | round trip p50 / p99 | 25 / 39 µs | 47 / 55 µs | 28 / 39 µs |
| 1 ms gap | 1 | round trip p50 / p99 | 15 / 41 µs | 39 / 91 µs | 41 / 109 µs |
| 1 ms gap | 4 | round trip p50 / p99 | 48 / 1 328 µs | 63 / 697 µs | 61 / 175 µs |
| 10 ms gap | 1 | round trip p50 / p99 | 56 / 184 µs | 655 / 1 134 µs | 681 / 950 µs |
| 10 ms gap | 4 | round trip p50 / p99 | 94 / 1 082 µs | 719 / 1 904 µs | 738 / 1 762 µs |

- **Back-to-back.** A ping per write costs the writer a zmq send and an I/O
  thread wake-up, so `arm-no` loses 23% (1 worker) to 43% (4 workers) of
  upstream's steps/s; upstream's `SNDHWM=1` drops most of those pings. Parked-
  only pings skip them while readers spin, and `arm-so` is within 9-11% of
  upstream. The rest is the adaptive grace bookkeeping, the blocked-wait
  metrics, and the fence and flag reads.
- **Paced.** A parked reader wakes through zmq and, on an idle core, through
  the CPU's idle-state exit. At a 10 ms gap that adds about 0.6 ms per round
  trip on this host. An earlier session of the same harness measured 0.11 ms
  more than upstream, so the figure depends on how deeply idle cores sleep. Raise
  `VLLM_SHM_BROADCAST_ADAPTIVE_MAX_GRACE_MS` to spin longer when step latency
  matters more than CPU.

No step in the matrix waited out the 5 s recheck; the largest round trip was
5.8 ms. Before the single-read fix in `_park`, `arm-so` stalled 5 s on 6-9% of
paced single-worker steps.

Rerun one variant with:

```bash
TMPDIR=/dev/shm/vbench PYTHONPATH=<tree> VLLM_TARGET_DEVICE=cpu \
  python benchmarks/benchmark_shm_broadcast.py --variant <name> --repeat 1 \
  --readers 1,4 --gaps 0,1,10 --idle-seconds 10 --target-wall 6 \
  --min-steps 500 --max-steps 600000 --json-out results.jsonl
```

`TMPDIR` must be short: zmq IPC paths are limited to about 107 characters.

## Formal models

`tools/formal/shm_broadcast/` holds three [Quint](https://quint-lang.org)
specifications:

- `message_queue.qnt` models one queue: the ring, the reader spin-or-park
  choice, the zmq channel with its high water mark, credits, command throttle
  and subscriptions, and the parked-flag protocol with store buffers on both
  flags, so a missing fence lets a store become visible late.
- `inference.qnt` composes four queues into one TP=2 executor loop: the RPC
  broadcast, two reply queues, and a TP `broadcast_object` inside every step.
  It abstracts the channel as lossless, losing a reader's first ping, or
  losing any ping.
- `paths.qnt` models one local and one remote reader with overflow payloads
  on `local_socket`, every message on `remote_socket`, and `shutdown()`
  stopping the writer and cancelling both readers.

The invariants:

- `noLostWakeup`: the wake-up invariant above.
- `ringSafe` / `inOrder`: every reader receives every message in order, and
  no slot is reused before it is read.
- `noLoss`: no data-socket message is dropped.
- `progress`: without timeouts, some process or I/O thread can always move
  until every message is read or, after shutdown, until every reader exits.

| Spec | Configuration | Result |
| --- | --- | --- |
| `message_queue` | `hwm1`, `hwm1OneReader`: `SNDHWM=1`, throttled commands | `noLostWakeup` fails |
| `message_queue` | `hwm1NoThrottle`: `SNDHWM=1`, commands on every send | holds |
| `message_queue` | `hwm2Slow`, `hwm4Slow`, `hwm4Burst`: finite high water mark | `noLostWakeup` fails |
| `message_queue` | `pr58917`, `unlimitedNoHandshake`: no subscription handshake | `noLostWakeup` fails |
| `message_queue` | `unlimitedHandshake`: lossless channel, ping every write | holds |
| `message_queue` | `skipIdleFenced`: lossless channel, parked-only pings | holds |
| `message_queue` | `skipIdleUnfenced`: parked-only pings without fences | `noLostWakeup` fails |
| `message_queue` | `skipIdleFlaglessPark`: a reader may poll without its flag | `noLostWakeup` fails |
| `message_queue` | `futexFenced`: futex wake of parked readers, full fences | holds |
| `message_queue` | `futexUnfenced`: futex wake without the fences | `noLostWakeup` fails |
| `inference` | `fixedDecode`, `fixedAllReply`: lossless channel | holds |
| `inference` | `hwm1Decode`, `raceDecode`: lossy channel | `noLostWakeup` and `progress` fail; the executor and both workers end parked |
| `inference` | `backpressureAtBound`: 1-slot rings, 3 outstanding RPCs | holds |
| `inference` | `backpressureOver`: 1-slot rings, 4 outstanding RPCs | `progress` fails |
| `paths` | `pathsCurrent`: default XPUB data sockets, uncancellable socket waits | `inOrder`, `noLoss` and `progress` fail |
| `paths` | `pathsNoDrop`: `XPUB_NODROP` only | `progress` fails: a reader blocked on a data socket never exits after shutdown |
| `paths` | `pathsCancel`: cancellable socket waits only | `inOrder`, `noLoss` and `progress` fail |
| `paths` | `pathsFixed`: `XPUB_NODROP` and cancellable socket waits | holds |

Run `tools/formal/shm_broadcast/check.sh` to check every configuration with
TLC and Apalache in parallel; it takes about 2 minutes on 16 cores and exits
non-zero when a result differs from the table. `check.sh tlc` or
`check.sh apalache` runs one backend. TLC covers the full state space of
every configuration. Apalache must find each expected violation within the
length of the TLC counterexample, and checks every behavior of up to 10 steps
(6 for `inference`) of each configuration that holds. Five configurations are
TLC-only because their counterexamples are too deep for Apalache: `hwm1`
(14 steps, 3-5 minutes), and `hwm2Slow`, `hwm4Slow`, `hwm4Burst` and
`backpressureOver` (20 to 32 steps, over an hour each). `hwm1OneReader` is the
Apalache-sized copy of `hwm1`.

The models treat a notification as a message that cannot overtake the
writer's written flag, as zmq's lock-based pipes and the kernel socket
guarantee, and they assume the payload copy is ordered before the written
flag by `memory_fence()`. They do not model the multi-node composition of the
inner-DP queues or the `READY` handshake on the data sockets.
