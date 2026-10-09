# iSWAP phase compensation in `iswap_RPE_f`

Record of the virtual-Z phase tracking implemented for the SNAIL-mediated
iSWAP gate, following section 4.3.2 of P. Lu's dissertation
(`43554_PLu_Final_Dissertation_v2.pdf`).

Implementation: [`utils.py`](utils.py) — `iswap_RPE_f`, `iswap_RPE_d`,
`process_tomography_schedule` / `process_tomography_pre_build`, and the shared
helpers `_detuning_residual`, `_iswap_rz_corrections`, `compute_iswap_t1`.

The RPE sequences follow Fig. 5 of Tiwari et al., *High-fidelity iSWAP gate
with Double Transmon Coupler* (arXiv 2604.27080), which parametrises the gate
as (its eq. 8)

```
iSWAP(theta_p, phi_p, theta_1, theta_2, phi_zz) =
  [[1, 0,                        0,                             0],
   [0, e^{i th2} cos(th_p),      i e^{i(th2+phi_p)} sin(th_p),  0],
   [0, i e^{i(th1-phi_p)} sin(th_p), e^{i th1} cos(th_p),       0],
   [0, 0,                        0,        e^{i(th1+th2+phi_zz)}]]
```

`iswap_RPE_d` is Fig. 5(d) and `iswap_RPE_f` is Fig. 5(f). Simulation of these
sequences gives the measured eigenphases as

| schedule | measures |
|---|---|
| `iswap_RPE_d` | `theta_1 + theta_2 + phi_zz` |
| `iswap_RPE_f` | `theta_1 - theta_2 - 2 phi_p` |

The `-2 phi_p` is the paper's own statement that "applying equal and opposite
Z rotations to each qubit frame is equivalent to shifting the parametric drive
phase by twice the same amount" — the same degeneracy `pump_phase` has here.

The correction rule below leaves `c_q0 + c_q1` exactly invariant, so it moves
only the `iswap_RPE_f` channel and provably cannot perturb what `iswap_RPE_d`
measures. Note also that `phi_zz` is the non-separable part of the `|11>`
phase: single-qubit Z rotations can null `theta_1` and `theta_2` but never
`phi_zz`, which has to go at the coupler bias.

---

## The correction rule

Each qubit carries a **correction register** `c_q`: the phase that must be
added to that qubit's NCO so its frame matches the actual excitation phase.
Both registers start at 0 at every `ResetClockPhase`.

An iSWAP occurring at time `t` after the phase reset transforms them as

```
c_a' = c_b + phi_0 - (Delta - Delta_tilde) * t
c_b' = c_a - phi_0 + (Delta - Delta_tilde) * t
```

and the gate actually inserted after the iSWAP is the **delta**:

```
Rz(q_a, c_a' - c_a)
Rz(q_b, c_b' - c_b)
```

### Derivation

At an iSWAP at time `t`:

- excitation on `a` has lab phase `omega_a * t + c_a`; on `b`, `omega_b * t + c_b`
- the iSWAP moves `b`'s excitation onto `a` and imprints the pump phase
  (eq. 53's off-diagonal `e^{i phi_s}`), so `a` now carries
  `omega_b * t + c_b + phi_s`
- but `a`'s NCO still reads `omega_a * t`, so the newly required register is
  `c_a' = c_b + phi_s - Delta * t`
- substituting `phi_s = Delta_tilde * t + phi_0` gives the rule above

---

## What each term covers

The dissertation lists four parts of two-qubit parametric-gate phase
compensation. The single rule above contains three of them.

| Part | Physical origin | Status |
|---|---|---|
| **1** — virtual phase `(phi_a - phi_b)` | virtual-Z gates already on the software books before the two-qubit gate | **Implemented** as the `c_a <-> c_b` swap (eq. 55) |
| **2** — `(Delta - Delta_tilde) * t1` | AC Stark shift during pumping detunes the qubits away from the pump frequency | **Implemented**, derived from the edge config |
| **3** — constant per-qubit AC Stark offset | fixed phase from pump ramp-up/down | **Exposed** as `pump_phase` (`phi_0`), default 0 |
| **4** — `\|G_a - G_b\| = G_s` LO mixing | hardware phase drift between independent generators | **Done in hardware** — pump LO derived by mixing the two qubit generators |

### Part 1 is inert on its own here

Part 1's `phi_a, phi_b` are specifically the **virtual-Z register**, not the
NCO's `freq x time` frame. The dissertation defines it explicitly:

> "There are potentially virtual-Z gates following the sequence, where we can
> record all virtual phase before two-qubit gates as `phi_a, phi_b`."

and keeps it distinct from what it calls the **real phase**,
`phi_a = (G_a + S_a) * T_0 + omega_a * t1`, which is what the NCO tracks.

In this schedule `index_to_operation` maps every Clifford to a *physical*
`Rxy` (`X180`, `X90`, `Y180`, `Y90`, `mX90`, `mY90`) — there are **no
virtual-Z gates anywhere in the sequence**. So `c_a = c_b = 0` and the part-1
swap has nothing to exchange until part 2 or `pump_phase` seeds the register.
Part 1 is not an independent knob; it is the propagation rule that carries
part 2's correction across each iSWAP.

### Part 2 is the term that does the work

`Delta - Delta_tilde` is a **small residual** (~100 kHz to ~1 MHz), *not* the
full ~260 MHz bare detuning. The pump tone imprints `Delta_tilde * t` on the
transferred excitation for free, cancelling almost all of the drift by itself.

An earlier version of this code used the full `360 * (f_q0 - f_q1) * delta_t`,
which over-corrected by a factor of ~300 and was hypersensitive to timing
(93 degrees of error per nanosecond).

### Part 4 is handled in hardware

The pump LO is derived by mixing the two qubit generators, so
`abs(G_a - G_b) = G_s` holds by construction and the generators' random phase
drift cancels — the dissertation's own solution, applied at the source rather
than corrected in software.

This is what makes the rest of the software tracking well posed. Because the
pump's phase is stable relative to the NCO resets from block to block, `phi_0`
is a genuine constant across the whole sweep rather than a per-block random
variable, and the `Delta_tilde * t` term the pump imprints on the transferred
excitation is coherent with the qubit frames. Without part 4 no amount of
virtual-Z bookkeeping would converge.

---

## Key implementation decisions

**Two-pass compile.** Pass 1 builds the schedule uncorrected purely to resolve
timing; `compute_iswap_t1` compiles it and reads the real iSWAP times out of
the timing table; pass 2 rebuilds with the computed `Rz` gates inserted. This
is safe because `Rz` compiles to a zero-duration `ShiftClockPhase`, so pass-1
timing remains valid in pass 2.

**Because the times come from the compiled table, `wait_time` (and any other
timing change) is compensated automatically** — no hardcoded gate durations.

**Reference point is `ResetClockPhase`, not `Reset`.** `Reset` has a ~250 us
thermalization duration and `ResetClockPhase` is scheduled at its *end*
(`ref_pt="end"`). Using `Reset`'s own start-time `abs_time` incorrectly folded
that entire thermalization wait into `t`.

Only `.01` clock resets count. `Measure` also emits `ResetClockPhase` on the
readout (`.ro`) clocks; counting those makes the shortest `.01` -> `.ro` gap
look like the block period.

**`t` runs to the RF turn-on, not the marker edge.** The marker is issued
`iSWAP_DELAY` early to compensate the pump's electrical delay, so the swap
happens at `marker + iSWAP_DELAY`:

```python
t = (marker_abs_time + iSWAP_DELAY) - reset_abs_time
```

`iSWAP_DELAY` is the *differential* path delay `d_pump - d_qubit`. The marker
is issued early so the pump lands in its intended slot on the qubit timeline,
and elapsed time in the qubit frame is `T_intended - T_reset`. The marker edge
itself is a hardware artifact.

Using the raw marker edge breaks down once `iSWAP_DELAY` exceeds whatever sits
between the reset and the gate: the marker then fires *before* its own block's
`ResetClockPhase`, the gate is anchored to the previous block, and `t` picks up
a whole `Reset`. This showed up in process tomography as corrections of
~72000 deg on identity-prep blocks (`t` = a full 253 us block period) while
normal blocks read ~8 deg.

**`t` is measured from the phase reset, not as the gap since the previous
iSWAP.** The mismatch is a frame offset set by *total elapsed phase*, not a
quantity that accumulates per gate.

**Two guards** in `compute_iswap_t1`, both added after the above failed
silently: any `t` at or beyond the block period raises, and any negative `t`
raises.

**iSWAP occurrences are found by port.** `CompositeiSWAPEdge` compiles the
gate down to a `MarkerPulse` on `"snail:switch"`, used as the fingerprint in
the timing table.

**The pump frequency is read from the device config**, not passed in — it
comes from the iSWAP edge the gate is actually pumped through:

```python
quantum_device.generate_device_config() \
    .edges["qubit1_qubit2"]["iSWAP"].factory_kwargs["pulse_frequency"]
```

So retuning the gate updates the correction on the next call, with no
argument to keep in sync. `_detuning_residual` tries both qubit orderings for
the edge name and raises if neither exists.

**Sign handling is internal.** `Delta_tilde` must carry the same sign as
`Delta = f01_q0 - f01_q1`, but the configured pump frequency is a positive
magnitude:

```python
detuning_tilde    = math.copysign(pump_freq, detuning)
detuning_residual = detuning - detuning_tilde
```

Subtracting the pump frequency directly would yield `-2 * pump_freq` whenever
`q0` sits below `q1`. Note also that `copysign(abs(Delta) - pump_freq, Delta)`
is **wrong** — it discards the inner sign when the pump sits above the bare
detuning.

---

## Applied to both RPE schedules

The rule lives in two shared helpers, `_detuning_residual` and
`_iswap_rz_corrections`, so the two schedules cannot drift apart.
`compute_iswap_t1` returns one flat chronological list; since block
boundaries follow from `lengths`, each schedule just consumes it
positionally with an iterator.

| schedule | `snail:switch` pulses per block | corrected |
|---|---|---|
| `iswap_RPE_f` | `2m` — all full iSWAPs | all of them |
| `iswap_RPE_d` | `m + 2` — sqrt-iSWAP, `m` full iSWAPs, sqrt-iSWAP | skip first and last |

Both check the total pulse count up front and raise rather than silently
mis-aligning the correction list against the gates if the schedule shape
ever changes.

### Why the sqrt-iSWAPs are skipped in `iswap_RPE_d`

- The **leading** one runs while both registers are still zero — the Clifford
  decomposition contains no virtual-Z gates — so there is nothing to exchange.
- The **trailing** one is followed only by the measurement, so no later gate
  needs a corrected frame, and the `m` full iSWAPs' corrections have already
  been applied before it.
- A software phase exchange is not valid for a **partial** swap anyway
  (eq. 56): for the iSWAP gate *family* only the pump-phase route holds,
  while the software-exchange route works only for the exact iSWAP.

---

## Validation

Two independent routes to `Delta - Delta_tilde`, using measured values:

```
f01 qubit1      = 4 507 760 421.045625 Hz
f01 qubit2      = 4 765 653 542.069289 Hz
pulse_frequency =   258 688 310        Hz   (iSWAP edge config)
Delta           =      -257.893121 MHz
```

Measured iSWAP times from the compiled timing table (ns), `iswap_RPE_f`:

```
[264, 906, 1548, 2190, ... , 18882]   ->  uniform 642 ns spacing
```

These are self-consistent with the schedule layout, which pins both gate
durations from two independent numbers:

```
t_first = 2 x (16 + D) + 2 x 32 = 264   ->  D = 100 ns  (single-qubit gate)
spacing = (G + 32) + (16 + D)   = 642   ->  G = 494 ns  (iSWAP)
```

| Source | `Delta - Delta_tilde` |
|---|---|
| hand-tuned 4.00 deg/gate (+180) | 794.885 kHz |
| computed from f01 + edge `pulse_frequency` | 795.189 kHz |
| **difference** | **304 Hz (0.04%)** |

A phase tuned by eye until the RPE looked right agrees to four significant
figures with a number derived from calibrated frequencies, the gate's own
configured pump frequency, and a compiled timing table.

That comparison was made at the earlier 643 ns spacing. At the current 642 ns
the model gives `183.784 deg` per spacing, i.e. `3.784 deg` mod 180.

### The 180 degree offset

Model gives `184.070 deg` per 643 ns spacing; the hand-tuned value was
`-4.00 deg`. The difference is exactly 180 degrees, which is the measurement's
own degeneracy.

That offset is almost certainly the iSWAP's intrinsic `i` factor —
eq. 53's off-diagonals are `i * sin(g_eff * t) * e^{+-i phi_s}`, and only
`phi_s` is modelled here. That is 90 degrees per transfer, 180 degrees across
the pair, constant per gate.

**Deliberately not implemented.** It would be a symmetric `+90` on both
registers (structurally different from `pump_phase`'s antisymmetric `+-phi_0`),
and it is unobservable here, so committing an unverified sign convention was
not worth it.

---

## Usage

```python
RPE_sched_kwargs = {
    "qubit_specifier": [qubit1, qubit2],
    "lengths": length,
    "acq_protocol": "ThresholdedAcquisition",
    "quantum_device": quantum_device,
}
```

Passing `quantum_device` is all that is needed — the pump frequency comes from
its edge config. `iswap_RPE_d` takes the same parameters, alongside its own
`sqrt_iswap_duration` and `wait_time`.

`quantum_device=None` (the default) means no compile pass and no `Rz`
inserted, byte-identical to the original uncorrected schedule.

`pump_phase` defaults to `0.0`.

Inspect the timings directly with:

```python
sched = iswap_RPE_f([qubit1, qubit2], lengths=[4], quantum_device=quantum_device)
print(compute_iswap_t1(sched, quantum_device))
```

### Process tomography

`process_tomography_schedule` compiles and runs internally, so computing the
correction inline would double the compile cost of every run (measured: 62 s
vs 33 s). The correction is therefore computed once by a separate call and
passed in:

```python
correction_phase = process_tomography_pre_build(
    [qubit1, qubit2], 5760, experiment_list, quantum_device)

results = process_tomography_schedule(
    [qubit1, qubit2], 5760, experiment_list,
    quantum_device=quantum_device, instrument_coordinator=ic,
    correction_phase=correction_phase)
```

The structural arguments must match between the two calls. A length mismatch
raises; a same-length mismatch cannot be detected, so re-run the pre-build
after any retune (it depends on `clock_freqs.f01()` and the edge pump
frequency). Omitting `correction_phase` on a schedule containing iSWAPs prints
a warning rather than failing silently.

---

## Caveats

**f01 reproducibility limits the accuracy.** qubit1's f01 was observed to move
169 kHz between two readings in a single calibration run, and spans
4.50776–4.51036 GHz across runs. The residual is a small difference of two
~4.6 GHz numbers, so the optimum drifts by roughly that much between
calibrations. Since the code recomputes from `clock_freqs.f01()` and the edge
config on every call, it tracks automatically — but a hardcoded residual would
go stale.

If a residual per-gate offset ever appears, its *sign pattern* identifies the
cause. A uniform timing-reference error and a `pump_phase` error both flip
sign every gate (the swap recursion carries `r_k` into `d_{k-1}` with
alternating sign); only an error in `Delta - Delta_tilde` keeps the same sign
across gates, with two unequal magnitudes.

**`operation_buffer_time` is applied twice for two-qubit gates** in the
original `index_to_operation`, once inside the subschedule and again via
`add_timing_constraint`. Harmless for the gate itself (16 ns of extra idle),
but it silently doubled any `wait_time` folded into the same argument, which
turned a ZZ measurement into a factor-of-2 error. Fixed by setting the
constraint's `rel_time` to 0 — the constraint is still needed to synchronise
against q1's chain, just not to add a second gap.

**Three compiles per call** when the correction is enabled — two inside
`iswap_RPE_f` plus `ScheduleGettable`'s own. Noticeable on large outer sweeps.

**`pump_phase` is degenerate with a constant per-gate `Rz`** at fixed iSWAP
spacing — it is the only free knob left, since the part-2 residual is now
derived rather than swept.
