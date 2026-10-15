// Model of vLLM shm_broadcast MessageQueue: one writer, local readers,
// an M-slot ring with written/read flags, and the SpinCondition notify
// channel (writer zmq PUB/XPUB -> per-reader SUB with CONFLATE).
//
// The reader's spin grace is arbitrary (fixed, adaptive, 0), so a reader may
// park whenever its slot is not ready.
//
// zmq channel per reader:
//   * The PUB app thread drops a message if the reader's subscription is not
//     yet known to it, or if its stale view says the pipe is full:
//     HWM > 0 and sent - seen >= HWM.
//   * The pipe reader (I/O side) drains messages and returns a credit every
//     LWM drained messages.
//   * The PUB app thread processes commands (subscriptions, credits) only on
//     a send, and at most once per throttle window. K bounds the number of
//     sends inside one window, so a refresh is optional on any send and
//     mandatory after K sends without one.
//
// Memory: the writer's written-flag store and a reader's parked-flag store
// each sit in a store buffer until a flush step makes them visible to the
// other process. A notification is a message: the writer's written flag is
// visible before its ping arrives (message passing through zmq and the
// kernel). Store-to-load ordering across the two flags exists only where
// FENCE puts a full fence (a native seq_cst fence; vLLM's lock-based
// memory_fence() does not provide it on weakly ordered CPUs).
//
// SKIP_IDLE models the parked-flag protocol. A reader stores parked = 1,
// fences, and re-checks its slot before polling; it clears the flag when it
// wakes. The writer stores the written flag, fences, and sends a ping only
// if some reader's parked flag is set.
module messageQueue {
  const READERS: Set[int]
  const M: int
  const MAXMSG: int
  const HWM: int
  const LWM: int
  const K: int
  // The writer waits for every notify subscription before its first send.
  const NOTIFY_HANDSHAKE: bool
  const SKIP_IDLE: bool
  const FENCE: bool
  // Bug variant: a reader may poll without setting its parked flag.
  const FLAGLESS_PARK: bool

  // ring buffer
  var written: int -> bool
  var seqOf: int -> int
  var readFlag: int -> (int -> bool)
  var ok: bool
  // writer
  var wSeq: int
  var wPc: str
  var wPend: int
  var sinceRefresh: int
  var drops: int
  // notify channel, per reader
  var subArrived: int -> bool
  var subscribed: int -> bool
  var sent: int -> int
  var inflight: int -> int
  var drained: int -> int
  var credit: int -> int
  var seen: int -> int
  var conflated: int -> bool
  // readers
  var rPc: int -> str
  var rSeq: int -> int
  var parked: int -> bool
  var parkPend: int -> bool

  pure val SLOTS = 0.to(M - 1)

  // Readers see only flushed written flags.
  def ready(r: int): bool = {
    val s = rSeq.get(r) % M
    rSeq.get(r) < MAXMSG and written.get(s) and not(readFlag.get(s).get(r))
  }

  def canSend(sub: int -> bool, sn: int -> int, r: int): bool =
    sub.get(r) and not(HWM > 0 and sent.get(r) - sn.get(r) >= HWM)

  action ringUnchanged = all {
    written' = written, seqOf' = seqOf, readFlag' = readFlag, ok' = ok,
  }
  action writerUnchanged = all {
    wSeq' = wSeq, wPc' = wPc, wPend' = wPend, sinceRefresh' = sinceRefresh,
    drops' = drops,
  }
  action chanUnchanged = all {
    subArrived' = subArrived, subscribed' = subscribed, sent' = sent,
    inflight' = inflight, drained' = drained, credit' = credit, seen' = seen,
    conflated' = conflated,
  }
  action readersUnchanged = all {
    rPc' = rPc, rSeq' = rSeq, parked' = parked, parkPend' = parkPend,
  }

  action init = all {
    written' = SLOTS.mapBy(_ => false),
    seqOf' = SLOTS.mapBy(_ => -1),
    readFlag' = SLOTS.mapBy(_ => READERS.mapBy(_ => false)),
    ok' = true,
    wSeq' = 0,
    wPc' = "acquire",
    wPend' = -1,
    // The first send always processes commands.
    sinceRefresh' = K,
    drops' = 0,
    subArrived' = READERS.mapBy(_ => false),
    subscribed' = READERS.mapBy(_ => false),
    sent' = READERS.mapBy(_ => 0),
    inflight' = READERS.mapBy(_ => 0),
    drained' = READERS.mapBy(_ => 0),
    credit' = READERS.mapBy(_ => 0),
    seen' = READERS.mapBy(_ => 0),
    conflated' = READERS.mapBy(_ => false),
    rPc' = READERS.mapBy(_ => "check"),
    rSeq' = READERS.mapBy(_ => 0),
    parked' = READERS.mapBy(_ => false),
    parkPend' = READERS.mapBy(_ => false),
  }

  // The reader's SUB connect and subscribe reach the writer's I/O thread.
  action subArrive(r: int): bool = all {
    not(subArrived.get(r)),
    subArrived' = subArrived.set(r, true),
    subscribed' = subscribed, sent' = sent, inflight' = inflight,
    drained' = drained, credit' = credit, seen' = seen, conflated' = conflated,
    ringUnchanged, writerUnchanged, readersUnchanged,
  }

  // Handshake variant: the writer receives every notify subscription
  // (processing commands) before it writes the first message.
  action writerHandshake = all {
    NOTIFY_HANDSHAKE,
    wSeq == 0,
    READERS.forall(r => subArrived.get(r)),
    not(READERS.forall(r => subscribed.get(r))),
    subscribed' = subArrived,
    subArrived' = subArrived, sent' = sent, inflight' = inflight,
    drained' = drained, credit' = credit, seen' = seen, conflated' = conflated,
    ringUnchanged, writerUnchanged, readersUnchanged,
  }

  // acquire_write + payload copy; the written-flag store is buffered.
  action writerAcquire = {
    val s = wSeq % M
    all {
      wPc == "acquire",
      wPend == -1,
      wSeq < MAXMSG,
      NOTIFY_HANDSHAKE implies READERS.forall(r => subscribed.get(r)),
      not(written.get(s)) or READERS.forall(r => readFlag.get(s).get(r)),
      written' = written.set(s, false),
      seqOf' = seqOf.set(s, wSeq),
      readFlag' = readFlag.set(s, READERS.mapBy(_ => false)),
      ok' = ok,
      wSeq' = wSeq + 1,
      wPc' = "notify",
      wPend' = s,
      sinceRefresh' = sinceRefresh,
      drops' = drops,
      chanUnchanged, readersUnchanged,
    }
  }

  // The writer's buffered written-flag store becomes visible.
  action writerFlush = all {
    wPend >= 0,
    written' = written.set(wPend, true),
    wPend' = -1,
    seqOf' = seqOf, readFlag' = readFlag, ok' = ok,
    wSeq' = wSeq, wPc' = wPc, sinceRefresh' = sinceRefresh, drops' = drops,
    chanUnchanged, readersUnchanged,
  }

  // One PUB send to every subscriber, optionally processing commands.
  action send(refresh: bool): bool = {
    val sub2 = if (refresh) subArrived else subscribed
    val seen2 = if (refresh) credit else seen
    all {
      refresh or sinceRefresh < K,
      subscribed' = sub2,
      seen' = seen2,
      sent' = READERS.mapBy(r =>
        if (canSend(sub2, seen2, r)) sent.get(r) + 1 else sent.get(r)),
      inflight' = READERS.mapBy(r =>
        if (canSend(sub2, seen2, r)) inflight.get(r) + 1 else inflight.get(r)),
      drops' = drops + READERS.filter(r => not(canSend(sub2, seen2, r))).size(),
      sinceRefresh' = if (refresh) 1 else sinceRefresh + 1,
      subArrived' = subArrived, drained' = drained, credit' = credit,
      conflated' = conflated,
    }
  }

  // notify() after the written flag. Without SKIP_IDLE it always sends, and
  // the ping cannot overtake the flag. With SKIP_IDLE it reads the parked
  // flags, after its fence when FENCE holds, and sends only if one is set.
  action writerNotify(refresh: bool): bool = all {
    wPc == "notify",
    if (SKIP_IDLE) (FENCE implies wPend == -1) else wPend == -1,
    if (SKIP_IDLE and READERS.forall(r => not(parked.get(r)))) all {
      subscribed' = subscribed, seen' = seen, sent' = sent,
      inflight' = inflight, drops' = drops, sinceRefresh' = sinceRefresh,
      subArrived' = subArrived, drained' = drained, credit' = credit,
      conflated' = conflated,
    } else send(refresh),
    wPc' = "acquire",
    wSeq' = wSeq,
    wPend' = wPend,
    ringUnchanged, readersUnchanged,
  }

  // The pipe reader drains one message into the reader's CONFLATE slot.
  action deliver(r: int): bool = {
    val d = drained.get(r) + 1
    all {
      inflight.get(r) > 0,
      inflight' = inflight.set(r, inflight.get(r) - 1),
      drained' = drained.set(r, d),
      credit' = if (LWM > 0 and d % (if (LWM > 0) LWM else 1) == 0) credit.set(r, d)
        else credit,
      conflated' = conflated.set(r, true),
      subArrived' = subArrived, subscribed' = subscribed, sent' = sent,
      seen' = seen,
      ringUnchanged, writerUnchanged, readersUnchanged,
    }
  }

  // acquire_read finds its slot ready (spinning or after a wake-up), reads
  // it and sets its read flag. It never consumes a notification.
  action readerRead(r: int): bool = {
    val s = rSeq.get(r) % M
    all {
      rPc.get(r) == "check",
      ready(r),
      ok' = (ok and seqOf.get(s) == rSeq.get(r)),
      readFlag' = readFlag.set(s, readFlag.get(s).set(r, true)),
      written' = written, seqOf' = seqOf,
      rSeq' = rSeq.set(r, rSeq.get(r) + 1),
      rPc' = rPc, parked' = parked, parkPend' = parkPend,
      writerUnchanged, chanUnchanged,
    }
  }

  // check() failed and the spin grace is gone. With SKIP_IDLE the reader
  // first stores its parked flag (buffered) and goes to the re-check.
  action readerPark(r: int): bool = all {
    rPc.get(r) == "check",
    rSeq.get(r) < MAXMSG,
    not(ready(r)),
    rPc' = rPc.set(r, if (SKIP_IDLE) "parking" else "park"),
    parkPend' = if (SKIP_IDLE) parkPend.set(r, true) else parkPend,
    rSeq' = rSeq, parked' = parked,
    ringUnchanged, writerUnchanged, chanUnchanged,
  }

  // The reader's buffered parked-flag store becomes visible.
  action readerFlushPark(r: int): bool = all {
    parkPend.get(r),
    parked' = parked.set(r, true),
    parkPend' = parkPend.set(r, false),
    rPc' = rPc, rSeq' = rSeq,
    ringUnchanged, writerUnchanged, chanUnchanged,
  }

  // Re-check after the parked store (after the fence when FENCE holds):
  // a ready slot cancels the park, otherwise enter poller.poll().
  action readerRecheck(r: int): bool = all {
    rPc.get(r) == "parking",
    FENCE implies not(parkPend.get(r)),
    if (ready(r)) all {
      rPc' = rPc.set(r, "check"),
      parked' = parked.set(r, false),
      parkPend' = parkPend.set(r, false),
    } else all {
      rPc' = rPc.set(r, "park"),
      parked' = parked,
      parkPend' = parkPend,
    },
    rSeq' = rSeq,
    ringUnchanged, writerUnchanged, chanUnchanged,
  }

  // poll() returns on a notification; recv() consumes the conflated one;
  // the reader clears its parked flag.
  action readerPoll(r: int): bool = all {
    rPc.get(r) == "park",
    conflated.get(r),
    conflated' = conflated.set(r, false),
    rPc' = rPc.set(r, "check"),
    parked' = parked.set(r, false),
    parkPend' = parkPend.set(r, false),
    rSeq' = rSeq,
    subArrived' = subArrived, subscribed' = subscribed, sent' = sent,
    inflight' = inflight, drained' = drained, credit' = credit, seen' = seen,
    ringUnchanged, writerUnchanged,
  }

  // SHM_READER_RECHECK_INTERVAL_MS expires: the 5 s fallback.
  action readerTimeout(r: int): bool = all {
    rPc.get(r) == "park",
    rPc' = rPc.set(r, "check"),
    parked' = parked.set(r, false),
    parkPend' = parkPend.set(r, false),
    rSeq' = rSeq,
    ringUnchanged, writerUnchanged, chanUnchanged,
  }

  // The bug the first parked-flag implementation had: the reader read its
  // spin grace twice, and when the grace expired between the reads it
  // polled without setting its parked flag.
  action readerParkFlagless(r: int): bool = all {
    SKIP_IDLE,
    FLAGLESS_PARK,
    rPc.get(r) == "check",
    rSeq.get(r) < MAXMSG,
    not(ready(r)),
    rPc' = rPc.set(r, "park"),
    rSeq' = rSeq, parked' = parked, parkPend' = parkPend,
    ringUnchanged, writerUnchanged, chanUnchanged,
  }

  action idle = all {
    ringUnchanged, writerUnchanged, chanUnchanged, readersUnchanged,
  }

  action step = any {
    writerAcquire,
    writerFlush,
    writerHandshake,
    writerNotify(true),
    writerNotify(false),
    idle,
    { nondet r = READERS.oneOf() any {
      subArrive(r),
      deliver(r),
      readerRead(r),
      readerPark(r),
      readerParkFlagless(r),
      readerFlushPark(r),
      readerRecheck(r),
      readerPoll(r),
      readerTimeout(r),
    } },
  }

  // A parked reader whose slot is ready always has a wake-up pending (or the
  // writer is between its written flag and its notify decision), so it
  // never depends on the recheck timeout.
  val noLostWakeup = READERS.forall(r =>
    (rPc.get(r) == "park" and ready(r))
      implies (conflated.get(r) or inflight.get(r) > 0 or wPc == "notify"))

  // Readers see messages in order, and the writer never reuses a slot that
  // a reader has not read.
  val ringSafe = ok
}

// SNDHWM=1 with the send throttle: the bug PR #58917 fixes.
module hwm1 {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 5, HWM = 1, LWM = 1,
    K = 3, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// hwm1 with one reader: the same lost wake-up in a smaller formula, so
// Apalache reaches it quickly.
module hwm1OneReader {
  import messageQueue(READERS = Set(0), M = 2, MAXMSG = 4, HWM = 1, LWM = 1,
    K = 3, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// SNDHWM=1 without the throttle: credits are always fresh.
module hwm1NoThrottle {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 5, HWM = 1, LWM = 1,
    K = 1, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// Finite HWM (zmq default 1000 scaled to 4, lwm = (hwm + 1) / 2), commands
// processed at least every second send.
module hwm4Slow {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 8, HWM = 4, LWM = 2,
    K = 2, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// Finite HWM, writer bursting every send inside one throttle window.
module hwm4Burst {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 8, HWM = 4, LWM = 2,
    K = 8, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// PR #58917 alone: finite HWM, no notify handshake.
module pr58917 {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 4, HWM = 4, LWM = 2,
    K = 2, NOTIFY_HANDSHAKE = false, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// SNDHWM=0 without the handshake: the startup race remains.
module unlimitedNoHandshake {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 4, HWM = 0, LWM = 0,
    K = 8, NOTIFY_HANDSHAKE = false, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// Finite HWM at the smallest size that still has credit lag (hwm 2,
// lwm 1): the same failure as hwm4Slow in fewer steps, for Apalache.
module hwm2Slow {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 6, HWM = 2, LWM = 1,
    K = 2, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// SNDHWM=0 plus the notify-subscription handshake, always notifying.
module unlimitedHandshake {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 8, HWM = 0, LWM = 0,
    K = 8, NOTIFY_HANDSHAKE = true, SKIP_IDLE = false, FENCE = true,
    FLAGLESS_PARK = false).*
}

// This change: lossless channel, pings only for parked readers, full fences.
module skipIdleFenced {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 6, HWM = 0, LWM = 0,
    K = 8, NOTIFY_HANDSHAKE = true, SKIP_IDLE = true, FENCE = true,
    FLAGLESS_PARK = false).*
}

// Parked-flag protocol without a store-to-load fence: the Dekker race.
module skipIdleUnfenced {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 4, HWM = 0, LWM = 0,
    K = 8, NOTIFY_HANDSHAKE = true, SKIP_IDLE = true, FENCE = false,
    FLAGLESS_PARK = false).*
}

// Parked-flag protocol, fenced, but a reader may poll without its flag.
module skipIdleFlaglessPark {
  import messageQueue(READERS = Set(0, 1), M = 2, MAXMSG = 4, HWM = 0, LWM = 0,
    K = 8, NOTIFY_HANDSHAKE = true, SKIP_IDLE = true, FENCE = true,
    FLAGLESS_PARK = true).*
}
