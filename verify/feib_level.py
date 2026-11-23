"""The checks of feib_bf.py, computed level by level, for larger n.

For a symmetric f the Fourier coefficient ft(S) depends only on s = |S|, so
every quantity in the paper is a sum over levels with multiplicity C(n, s).
This makes n in the tens reachable.  Everything is computed in mpmath at
DPS digits, because I(f) can be as small as about p^(n-1).

Profiles: for n <= ALLMAX every non-constant profile; above it every
single-jump profile, and seeded random samples of two-jump, three-jump and
unrestricted profiles.  p runs over a grid in (0, 1/2], with p = 1e-3 and
p = 1e-6 added for n <= SMALLMAX; the reduction from p > 1/2 is checked mask
by mask in feib_bf.py.

Checked (inequalities with margin 0; identities to 1e-30 relative, (A1) to
1e-20 absolute; B^+ <= H^raw to 1e-20):
  Theorem A(1)   I = n p_eff, sum mu_m = 1, W_s = I mu_{s-1}/s, formula (A1);
  Theorem A(3)   v <= e^{-M}(I_0(M) + I_1(M)) < min(1, sqrt(2/(pi M)));
  Lemma tilt     at lambda = 1/(nv);
  Lemma dich     the jump k chosen as in the proof, with v <= 1/d(k) if k != k0;
  (master)       E and E^raw against the right-hand sides of eq:master;
  Lemma poi      (gibbs) at alpha = 7/2 and alpha = 4, and the two closed-form
                 bounds on Q(M; alpha) at these alpha for every M that occurs;
  Remark or      for f = OR_n: the exact value of H^nz/I, V and I as displayed;
  Theorem B      (B1)-(B3), and B^+ <= H^raw.
Run:  python feib_level.py [n1,n2,...] [samples]
"""
import itertools
import math
import random
import sys

from mpmath import mp, mpf, log, sqrt, exp, binomial, pi

mp.dps = 350
NS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10, 16, 24, 32, 48, 64]
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 150
ALLMAX = 10
PS = ["0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]
PSMALL = ["0.001", "0.000001"]
SMALLMAX = 32
L2 = log(2)
REL = mpf(10) ** -30


def lg2(x):
    return log(x) / L2


def level_matrix(n, p):
    """A[s][k] with ft(level s) = sum_k g_k A[s][k]."""
    q = 1 - p
    r = sqrt(p * q)
    a, b = q / r, -p / r
    A = []
    for s in range(n + 1):
        row = []
        for k in range(n + 1):
            inner = mpf(0)
            for j in range(max(0, k - (n - s)), min(s, k) + 1):
                inner += math.comb(s, j) * math.comb(n - s, k - j) * a ** j * b ** (s - j)
            row.append(p ** k * q ** (n - k) * inner)
        A.append(row)
    return A


from functools import lru_cache


@lru_cache(maxsize=None)
def bessel_sum(M):
    s0 = s1 = mpf(0)
    t0, t1 = mpf(1), M / 2
    for k in range(400):
        s0 += t0
        s1 += t1
        t0 *= (M / 2) ** 2 / ((k + 1) ** 2)
        t1 *= (M / 2) ** 2 / ((k + 1) * (k + 2))
    return exp(-M) * (s0 + s1)


@lru_cache(maxsize=None)
def Qpoi(M, a, terms=600):
    b = 1 - log(a)
    s, logfac = mpf(0), mpf(0)
    for m in range(terms):
        logfac += log(m + 1)
        lt = m * log(M) - logfac + (m + 1) * (log(m + 1) - log(m + 1 + a * M)) - (m + 1) * b
        s += exp(lt)
    return s


@lru_cache(maxsize=None)
def q_bounds(M, a):
    """The two bounds on Q(M; alpha) stated in Lemma poi."""
    Q = Qpoi(M, a)
    assert Q <= (exp(a * M / exp(1)) - 1) / (M * (1 + a * M)), ("Q bound 1", float(M), float(a))
    if M >= 1 / (2 * a):
        assert Q < 3 * sqrt(pi) / 8 * mpf(2) ** mpf("0.25") * a ** mpf("0.25") * M ** mpf("-0.75"), \
            ("Q bound 2", float(M), float(a))
    return True


def Cint(a):
    return log(1 + a + sqrt(a * (a + 2))) + 1 - log(a)


def pmf(N, k, p):
    return binomial(N, k) * p ** k * (1 - p) ** (N - k)


def dich_k(n, J, p):
    q = 1 - p
    N = n - 1
    k0 = int(math.floor(n * p))
    if k0 + 1 in J:
        return k0, k0
    lo = [r for r in J if r <= k0]
    hi = [r for r in J if r >= k0 + 2]
    cand = []
    if lo:
        r = max(lo)
        t = sum(pmf(n, j, p) for j in range(0, r))
        cand.append((t / (p * q * n * pmf(N, r - 1, p)), r))
    if hi:
        r = min(hi)
        t = sum(pmf(n, j, p) for j in range(r, n + 1))
        cand.append((t / (p * q * n * pmf(N, r - 1, p)), r))
    ratio, r = max(cand)
    return r - 1, k0


def profiles(n, rng):
    if n <= ALLMAX:
        for gb in itertools.product([-1, 1], repeat=n + 1):
            if gb[0] == 1 and len(set(gb)) > 1:
                yield gb
        return
    def from_jumps(J):
        g, c = [], 1
        for k in range(n + 1):
            if k in J:
                c = -c
            g.append(c)
        return tuple(g)
    for r in range(1, n + 1):
        yield from_jumps({r})
    for size in (2, 3):
        for _ in range(SAMPLES):
            yield from_jumps(set(rng.sample(range(1, n + 1), size)))
    for _ in range(SAMPLES):
        g = [1] + [rng.choice([-1, 1]) for _ in range(n)]
        if len(set(g)) > 1:
            yield tuple(g)


orcount = [0]


def check(n, p, g, A, worst, regimes, tag):
    q = 1 - p
    N = n - 1
    c = [sum(g[k] * A[s][k] for k in range(n + 1)) for s in range(n + 1)]
    w = [x * x for x in c]
    m = [math.comb(n, s) for s in range(n + 1)]
    J = [r for r in range(1, n + 1) if g[r - 1] != g[r]]
    piJ = sum(pmf(N, r - 1, p) for r in J)
    peff = 4 * p * q * piJ
    I = sum(s * m[s] * w[s] for s in range(n + 1))
    V = sum(m[s] * w[s] for s in range(1, n + 1))
    assert abs(I - n * peff) <= REL * I, ("A1 I", tag)
    assert abs(w[0] + V - 1) <= REL, ("Parseval", tag)
    floor = REL * I   # weights below this are cancellation noise or negligible
    live = [s for s in range(1, n + 1) if m[s] * w[s] > floor * mpf(10) ** -40]
    Hnz = sum(m[s] * w[s] * lg2(1 / w[s]) for s in live)
    H0 = (1 - V) * lg2(1 / (1 - V)) if V < 1 else mpf(0)
    H = H0 + Hnz
    Hraw = Hnz - I * lg2(1 / (4 * p * q))
    Bp = mpf(0)
    for s in live:
        up = sum(math.comb(n - s, t - s) * w[t] for t in range(s, n + 1))
        Bp += m[s] * w[s] * lg2(up / w[s])
    res = {"B1": H / I - lg2(1 / (p * q)), "Hraw": Hraw / I, "H0+Hraw": (H0 + Hraw) / I,
           "B+": Bp / I, "B": (H0 + Bp) / I, "B+-Hraw": (Bp - Hraw) / I}
    bounds = {"B1": 4.25, "Hraw": 5.75, "H0+Hraw": 6.25, "B+": 5.75, "B": 6.25, "B+-Hraw": 1e-20}
    for key, val in res.items():
        assert val < bounds[key], (key, tag, float(val))
        if val > worst.get(key, (-99,))[0]:
            worst[key] = (float(val), n, float(p), J)
    if n == 1:
        return 6
    M = N * p * q
    v = V / I
    mu = [math.comb(N, k) * w[k + 1] / peff for k in range(n)]
    assert abs(sum(mu) - 1) <= REL * 10, ("A1 mu", tag)
    for s in range(1, n + 1):
        assert abs(m[s] * w[s] - I * mu[s - 1] / s) <= REL * I, ("A1 W_s", tag)
    a1 = sum(mu[k] / (k + 1) * log(math.comb(N, k) / (peff * mu[k])) for k in range(n) if (k + 1) in live) / L2
    assert abs(a1 - Hnz / I) <= mpf(10) ** -20, ("A1", tag, float(a1), float(Hnz / I))
    vB = bessel_sum(M)
    assert v <= vB and vB < min(1, sqrt(2 / (pi * M))), ("A3", tag)
    assert n * v >= 1 - REL
    if n * v > 1 + REL:
        lam = 1 / (n * v)
        tilt = (1 - v) * lg2(1 / lam) + (n * v - 1) * lg2(1 / (1 - lam)) + v * lg2(1 / (n * lam * v * peff))
    else:
        tilt = v * lg2(1 / peff)
    assert Hnz / I <= tilt + REL, ("tilt", tag)
    k, k0 = dich_k(n, J, p)
    assert k + 1 in J and piJ >= pmf(N, k, p), ("dich k", tag)
    vub = min(mpf(1), vB)
    if k != k0:
        d = n * p - k if k < k0 else k + 1 - n * p
        assert d >= 1 and v <= 1 / d, ("dich", tag, k, float(v), float(d))
        vub = min(vub, 1 / d)
    plb = 4 * p * q * pmf(N, k, p)
    sg = min(vub, 1 / (n * plb))
    E = H / I - lg2(1 / (p * q))
    tail = (1 - sg) * lg2(n * sg * p * q) + sg * lg2(1 / pmf(N, k, p))
    assert E < (1 + sg) / L2 - 2 * sg + tail, ("master E", tag)
    assert Hraw / I < 1 / L2 + 2 - 2 * sg + tail, ("master Eraw", tag)
    for a in (mpf("3.5"), mpf(4)):
        q_bounds(M, a)
        rhs = lg2(1 / (p * q)) + (Cint(a) + v * log(Qpoi(M, a) / (4 * v * piJ))) / L2
        assert Hnz / I <= rhs, ("gibbs", float(a), tag)
    if J == [1]:
        # Remark or (the value of g at 0 differs from the rest)
        qn = q ** n
        ex = lg2(1 / p) + (2 * (1 - qn) / p - 1) * lg2(1 / q) - 2 * (1 - qn) / (n * p)
        assert abs(Hnz / I - ex) <= mpf(10) ** -20, ("OR value", tag, float(Hnz / I), float(ex))
        assert abs(V - (1 - (1 - 2 * qn) ** 2)) <= REL and abs(I - 4 * n * p * qn) <= REL * I, ("OR V, I", tag)
        orcount[0] += 1
    if k == 0:
        regimes[1] += 1
    elif k == N:
        regimes[2] += 1
    elif k <= N * p / 2:
        regimes[3] += 1
    elif k > 3 * N * p / 2:
        regimes[4] += 1
    else:
        regimes[5] += 1
    return 6 + 9


def main():
    rng = random.Random(20261123)
    worst, regimes, checks, nprof = {}, [0] * 6, 0, 0
    for n in NS:
        for ps in PS + (PSMALL if n <= SMALLMAX else []):
            p = mpf(ps)
            A = level_matrix(n, p)
            for g in profiles(n, rng):
                checks += check(n, p, g, A, worst, regimes, (n, ps, g))
                nprof += 1
        print(f"n={n} done, profiles so far {nprof}", flush=True)
    print(f"n in {NS}, p in {PS} (and {PSMALL} for n <= {SMALLMAX}): {nprof} profiles, {checks} checks passed")
    print("OR_n checked", orcount[0], "times")
    print("regime counts 1..5:", regimes[1:])
    for key, val in worst.items():
        print(f"  largest {key:8s} {val[0]:>10.5f}  at n={val[1]}, p={val[2]}, jumps={val[3]}")


main()
