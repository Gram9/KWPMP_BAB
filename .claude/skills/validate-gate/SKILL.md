---
name: validate-gate
description: Run a validation gate, record the result, and update the status table
disable-model-invocation: true
---

Run validation gate $ARGUMENTS and record the outcome.

`disable-model-invocation: true` — this has side effects (writes to docs and the
worklog) so it is manual-only. Invoke with `/validate-gate 1`.

## Steps

1. **Read the gate definition** in `docs/02_validation_gates.md`. Do not proceed
   from memory of what the gate does — read the current text, since gates get
   refined.

2. **Confirm prerequisites.** Gates are ordered. If an earlier gate is not
   marked PASS, stop and say so rather than running this one. Gate 4
   specifically must not be attempted before Gates 1–3 pass.

3. **Run it.** Execute the gate's test. Show the actual command and its actual
   output — never assert a pass without the evidence in the transcript.

4. **Report against the stated threshold**, not against a vibe. Gates have
   numeric criteria (Gate 2: correlation > 0.99; Gate 4: correlation > ~0.9
   against AQR's published BAB). State the number achieved next to the number
   required.

5. **On failure, diagnose in the documented order.** For Gate 4 that is:
   universe definition → delisting returns → beta estimator windows/minimums →
   weighting scheme → leg scaling. Check the realized market loading first — if
   it isn't near zero the estimator is the problem, not the portfolio
   construction. Do not start editing code until you have named a hypothesis.

6. **Before recording any PASS, prove the check can fail.** A passing number
   is not evidence unless the check that produced it is capable of failing.
   Two gates in this project were marked PASS and later retracted because the
   statistic was blind to the defect underneath it — correlation is location-
   and scale-invariant, so a +50bp/day bias and a 1.50x scale error both score
   1.0000000000.

   Run the `gate-verifier` subagent, then inject the error class the gate
   exists to catch and confirm the assertion fires. If the gate has only
   correlation or ratio-based assertions, it is INSUFFICIENT regardless of the
   number achieved — say so and stop rather than recording a PASS.

7. **Update `docs/02_validation_gates.md`** — set the status, and if it passed,
   record the number achieved so a future regression is detectable. Record the
   injected-error demonstration alongside it; a PASS without one is a claim,
   not a result.

8. **Append to `docs/worklog.md`**: date, gate, result, key number, and anything
   surprising. Two lines is enough.

## Rules

- **Never mark a gate PASS without pasted evidence.** "Tests pass" is not
  evidence; the test output is.
- **Never loosen a threshold to make a gate pass.** If a threshold seems wrong,
  say so and stop — changing the bar to clear it defeats the purpose of having
  gates at all.
- If the gate reveals a data-layer finding, add it to `docs/01_data_notes.md`
  too. That file is the project's institutional memory.