# The p-biased Fourier entropy–influence inequality for symmetric Boolean functions

Yitzchak Shmalo

Let μ_p be the p-biased product measure on {0,1}^n. Every non-constant symmetric Boolean function f satisfies

    H(f) < (log2(1/(p(1-p))) + 4.25) · I(f)

for every p in (0,1) and every n, where H is the Fourier entropy and I the total influence with respect to μ_p. The coefficient of log2(1/(p(1-p))) cannot be lowered, as the OR function shows when p → 0. After the level-by-level term I(f)·log2(1/(4p(1-p))) is removed, the remaining entropy is less than 6.25·I(f), and both its part on non-empty sets and the upward-cancellation functional B⁺(f) are less than 5.75·I(f).

## Contents

- `main.tex`, `main.pdf`: the paper.
- `verify/certify.py`: interval-arithmetic certificate of every constant used in the proof (mpmath `iv`, outward rounding). Output in `verify/certify_output.txt`.
- `verify/feib_bf.py`: numerical checks of Theorem A(1), A(3), Theorem B and the lemmas on every non-constant symmetric function with n ≤ 8, for ten values of p. Output in `verify/feib_bf_output.txt`.
- `verify/feib_level.py`: the same checks level by level in high-precision arithmetic on selected functions with n ≤ 96. Output in `verify/feib_level_output.txt`.

The numerical checks are not used in the proofs.

## Reproducing the outputs

Requires Python 3 and mpmath. From `verify/`:

    python certify.py 400 1200
    python feib_bf.py
    python feib_level.py 10,16,24,32,48,64,96

Each command prints the corresponding output file.
