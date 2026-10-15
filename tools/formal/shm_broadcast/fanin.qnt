// Model of the executor's multi-queue reply wait (MessageQueue.wait_many):
// a fan-in reader snapshots the readers' futex park words of several reply
// queues, then sleeps in one FUTEX_WAITV until one of the watched words
// changes or the recheck timeout expires.
//
// Per queue: writer publishes the message -> the parked reader is woken by
// the writer's exchange-and-FUTEX_WAKE -> the reader consumes the message and
// clears its word. The fan-in reader must never sleep while some queue holds
// a published message with no wake source it can still receive.
//
// Kernel futex semantics used: FUTEX_WAIT compares the word with 1 under the
// kernel lock, so a wake that lands between the snapshot and the sleep makes
// the sleep return immediately.
//
// In the real protocol the actors are separate processes (a writer and a
// reader per queue, the fan-in reader in the executor). They are merged into
// one interleaving here, which over-approximates their asynchrony.
module fanin {
  const QUEUES: Set[int]
  const MAXMSG: int
  // Models a writer wake that skips both the futex and the ping when the
  // word reads 0, racing the reader into its park. Must be rejected.
  const BROKEN_WAKE: bool

  // per queue: 0 = no message, 1 = published unread, 2 = being read
  var msg: int -> int
  var parked: int -> bool
  var notifyPending: int -> bool
  var wPc: int -> str
  // fan-in reader
  var exPc: str
  var snap: int -> int  // -1: not watching, 0: reader awake at snapshot, 1: parked

  def ready(q: int): bool = msg.get(q) == 1

  action init = all {
    msg' = QUEUES.mapBy(_ => 0),
    parked' = QUEUES.mapBy(_ => false),
    notifyPending' = QUEUES.mapBy(_ => false),
    wPc' = QUEUES.mapBy(_ => "publish"),
    exPc' = "awake",
    snap' = QUEUES.mapBy(_ => -1),
  }

  // ---- writer ----
  action writerPublish(q: int): bool = all {
    wPc.get(q) == "publish",
    msg.get(q) == 0,
    msg' = msg.set(q, 1),
    wPc' = wPc.set(q, "wake"),
    parked' = parked, notifyPending' = notifyPending,
    exPc' = exPc, snap' = snap,
  }

  action writerWake(q: int): bool = all {
    wPc.get(q) == "wake",
    if (parked.get(q)) all {
      parked' = if (BROKEN_WAKE) parked else parked.set(q, false),
      notifyPending' = notifyPending,
    } else all {
      parked' = parked,
      notifyPending' = notifyPending.set(q, true),
    },
    wPc' = wPc.set(q, "publish"),
    msg' = msg, exPc' = exPc, snap' = snap,
  }

  // ---- per-queue reader ----
  // A reader with a message ready consumes it; it never parks over one.
  action readerPark(q: int): bool = all {
    not(parked.get(q)),
    msg.get(q) != 1,
    parked' = parked.set(q, true),
    msg' = msg, notifyPending' = notifyPending, wPc' = wPc,
    exPc' = exPc, snap' = snap,
  }

  // The reader wakes on its futex or on the notify ping, clears its word,
  // and consumes the message.
  action readerRead(q: int): bool = all {
    parked.get(q) and (ready(q) or notifyPending.get(q)),
    parked' = parked.set(q, false),
    notifyPending' = notifyPending.set(q, false),
    msg' = msg.set(q, 2),
    wPc' = wPc, exPc' = exPc, snap' = snap,
  }

  action readerFinish(q: int): bool = all {
    msg.get(q) == 2,
    msg' = msg.set(q, 0),
    parked' = parked, notifyPending' = notifyPending, wPc' = wPc,
    exPc' = exPc, snap' = snap,
  }

  // ---- fan-in reader (the executor) ----
  action exSnapshot: bool = all {
    exPc == "awake",
    snap' = QUEUES.mapBy(q => if (parked.get(q)) 1 else 0),
    exPc' = "park",
    msg' = msg, parked' = parked, notifyPending' = notifyPending, wPc' = wPc,
  }

  // FUTEX_WAITV returns when a watched word flips or a ping is pending.
  action exWoke: bool = all {
    exPc == "park",
    QUEUES.exists(q =>
      snap.get(q) == 1 and (not(parked.get(q)) or notifyPending.get(q)))
      or QUEUES.exists(q => ready(q) and notifyPending.get(q)),
    exPc' = "awake",
    msg' = msg, parked' = parked, notifyPending' = notifyPending, wPc' = wPc,
    snap' = snap,
  }

  // The recheck timeout.
  action exTimeout: bool = all {
    exPc == "park",
    exPc' = "awake",
    msg' = msg, parked' = parked, notifyPending' = notifyPending, wPc' = wPc,
    snap' = snap,
  }

  action step = any {
    { nondet q = QUEUES.oneOf() any {
      writerPublish(q), writerWake(q), readerPark(q), readerRead(q),
      readerFinish(q)
    } },
    exSnapshot, exWoke, exTimeout,
  }

  // While the fan-in reader is parked, every published message must have a
  // wake source it can still receive: a watched park word, a pending ping,
  // or an awake reader that will consume the message.
  val noStuckWait = QUEUES.forall(q =>
    (exPc == "park" and ready(q))
      implies ((snap.get(q) == 1 and parked.get(q))
        or notifyPending.get(q)
        or wPc.get(q) == "wake"
        or not(parked.get(q))))
}

module faninTwo {
  import fanin(QUEUES = Set(0, 1), MAXMSG = 3, BROKEN_WAKE = false).*
}

module faninFour {
  import fanin(QUEUES = Set(0, 1, 2, 3), MAXMSG = 2, BROKEN_WAKE = false).*
}

// A writer wake that skips the futex when the word reads 0 must be caught.
module faninBroken {
  import fanin(QUEUES = Set(0, 1), MAXMSG = 3, BROKEN_WAKE = true).*
}
