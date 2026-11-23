"""Exhaustive checks of the statements of the paper on small cubes.

For every n <= NMAX, every p in a grid and every non-constant symmetric f on
{0,1}^n, the Fourier weights are computed by a level formula in 50-digit
arithmetic; for n <= NDIRECT they are also computed from the definition and
compared.  For p > 1/2 the weights are computed both at (f, p) and at the
reduced pair (f(1 - x), 1 - p) and compared mask by mask, which checks the
reduction used in the paper; the remaining checks then run at p <= 1/2.
The following are checked in floating point (tolerance 1e-9):
  Theorem A(1)   I = n p_eff, sum mu_m = 1, W_s = I mu_{s-1}/s, formula (A1);
  Theorem A(3)   v <= e^{-M}(I_0(M) + I_1(M));
  Lemma tilt     at lambda = 1/(nv);
  Lemma dich     the jump k chosen as in the proof, with v <= 1/d(k) if k != k0;
  (master)       E and E^raw against the right-hand sides of eq:master;
  (gibbs)        Lemma poi at alpha = 7/2 and alpha = 4;
  Theorem B      the five inequalities (B1)-(B3), and B^+ <= H^raw.
It also counts how often each of the five regimes occurs.
Run:  python feib_bf.py [NMAX]
"""
import itertools
import math
import sys

from mpmath import mp, mpf, sqrt as msqrt

mp.dps = 50

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NDIRECT = 5   # up to this n the level formula is compared with the definition
PS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.95]
TOL = 1e-9
L2 = math.log(2)


def log2(x):
    return math.log(x) / L2


def level_coeffs(n, g, p):
    """ft(S) for |S| = s (it depends only on s), by counting x with |x| = k, |x & S| = j."""
    p = mpf(p)
    q = 1 - p
    r = msqrt(p * q)
    out = []
    for s in range(n + 1):
        c = mpf(0)
        for k in range(n + 1):
            inner = mpf(0)
            for j in range(max(0, k - (n - s)), min(s, k) + 1):
                inner += math.comb(s, j) * math.comb(n - s, k - j) * (q / r) ** j * (-p / r) ** (s - j)
            c += g[k] * p ** k * q ** (n - k) * inner
        out.append(c)
    return out


def weights(n, g, p):
    """Squared Fourier coefficients keyed by bitmask, from the level formula (floats)."""
    c = level_coeffs(n, g, p)
    return {S: float(c[bin(S).count("1")] ** 2) for S in range(1 << n)}


def weights_direct(n, g, p):
    """Squared Fourier coefficients ft(S)^2 of f(x) = g(|x|), keyed by S as a bitmask."""
    q = 1 - p
    pts = range(1 << n)
    w = {}
    for S in range(1 << n):
        c = 0.0
        for x in pts:
            k = bin(x).count("1")
            m = (p ** k) * (q ** (n - k))
            b = 1.0
            for i in range(n):
                if S >> i & 1:
                    b *= ((x >> i & 1) - p) / math.sqrt(p * q)
            c += m * g[k] * b
        w[S] = c * c
    return w


def bessel_sum(M):
    """e^{-M}(I_0(M) + I_1(M)) by its power series."""
    s0 = s1 = 0.0
    t0, t1 = 1.0, M / 2
    for k in range(200):
        s0 += t0
        s1 += t1
        t0 *= (M / 2) ** 2 / ((k + 1) ** 2)
        t1 *= (M / 2) ** 2 / ((k + 1) * (k + 2))
    return math.exp(-M) * (s0 + s1)


def Qpoi(M, a, terms=400):
    b = 1 - math.log(a)
    s, logfac = 0.0, 0.0
    for m in range(terms):
        logfac += math.log(m + 1)
        lt = m * math.log(M) - logfac + (m + 1) * (math.log(m + 1) - math.log(m + 1 + a * M)) - (m + 1) * b
        s += math.exp(lt)
    return s


def Cint(a):
    return math.log(1 + a + math.sqrt(a * (a + 2))) + 1 - math.log(a)


def binom_pmf(N, k, p):
    return math.comb(N, k) * p ** k * (1 - p) ** (N - k)


def dich_k(n, g, p):
    """The k of Lemma dich, chosen as in its proof (p <= 1/2)."""
    q = 1 - p
    N = n - 1
    J = [r for r in range(1, n + 1) if g[r - 1] != g[r]]
    k0 = math.floor(n * p)
    if k0 + 1 in J:
        return k0, k0, J
    lo = [r for r in J if r <= k0]
    hi = [r for r in J if r >= k0 + 2]
    cand = []
    tau_m = lambda r: sum(math.comb(n, j) * p ** j * q ** (n - j) for j in range(0, r))
    tau_p = lambda r: sum(math.comb(n, j) * p ** j * q ** (n - j) for j in range(r, n + 1))
    if lo:
        r = max(lo)
        cand.append((tau_m(r) / (p * q * n * binom_pmf(N, r - 1, p)), r))
    if hi:
        r = min(hi)
        cand.append((tau_p(r) / (p * q * n * binom_pmf(N, r - 1, p)), r))
    ratio, r = max(cand)
    return r - 1, k0, J


def main():
    worst = {}
    regimes = [0] * 6
    checks = 0
    for n in range(1, NMAX + 1):
        N = n - 1
        for p0 in PS:
            for gb in itertools.product([-1, 1], repeat=n + 1):
                if len(set(gb)) == 1 or gb[0] == -1:
                    continue
                # reduction to p <= 1/2: (f, p) -> (f(1 - x), q)
                if p0 > 0.5:
                    p, g = 1 - p0, tuple(reversed(gb))
                else:
                    p, g = p0, gb
                q = 1 - p
                w = weights(n, g, p)
                if p0 > 0.5:
                    w0 = weights(n, gb, p0)
                    assert all(abs(w0[S] - w[S]) <= 1e-12 * max(w0[S], w[S]) + 1e-60 for S in w), \
                        ("reduction", n, p0, gb)
                    checks += 1
                if n <= NDIRECT:
                    wd = weights_direct(n, g, p)
                    assert all(abs(w[S] - wd[S]) < 1e-12 for S in w), ("level formula", n, p, g)
                V = sum(x for S, x in w.items() if S)   # not 1 - w[0]: that cancels when V is tiny
                I = sum(bin(S).count("1") * x for S, x in w.items())
                Hnz = sum(x * log2(1 / x) for S, x in w.items() if S and x > 1e-300)
                H0 = (1 - V) * log2(1 / (1 - V)) if V < 1 else 0.0
                H = H0 + Hnz
                Hraw = Hnz - I * log2(1 / (4 * p * q))
                Bp = 0.0
                for S, x in w.items():
                    if S and x > 1e-300:
                        up = sum(y for T, y in w.items() if T & S == S)
                        Bp += x * log2(up / x)
                res = {"B1": H / I - log2(1 / (p * q)), "Hraw": Hraw / I, "H0+Hraw": (H0 + Hraw) / I,
                       "B+": Bp / I, "B": (H0 + Bp) / I, "B+-Hraw": (Bp - Hraw) / I}
                bounds = {"B1": 4.25, "Hraw": 5.75, "H0+Hraw": 6.25, "B+": 5.75, "B": 6.25, "B+-Hraw": TOL}
                for key, val in res.items():
                    assert val < bounds[key], (key, n, p0, gb, val)
                    if val > worst.get(key, (-99,))[0]:
                        worst[key] = (round(val, 5), n, p0, gb)
                checks += 6
                if n == 1:
                    continue
                M = N * p * q
                v = V / I
                J = [r for r in range(1, n + 1) if g[r - 1] != g[r]]
                piJ = sum(binom_pmf(N, r - 1, p) for r in J)
                peff = 4 * p * q * piJ
                # Theorem A(1): mu_m from D_1 f, i.e. ft(S)^2 over S containing coordinate 0
                mu = [0.0] * n
                for S, x in w.items():
                    if S & 1:
                        mu[bin(S).count("1") - 1] += x
                mu = [x / peff for x in mu]
                assert abs(I - n * peff) < TOL and abs(sum(mu) - 1) < TOL
                for s in range(1, n + 1):
                    Ws = sum(x for S, x in w.items() if bin(S).count("1") == s)
                    assert abs(Ws - I * mu[s - 1] / s) < TOL, ("A1 W_s", n, p, g)
                a1 = sum(mu[m] / (m + 1) * math.log(math.comb(N, m) / (peff * mu[m]))
                         for m in range(n) if mu[m] > 1e-300) / L2
                assert abs(a1 - Hnz / I) < 1e-7, ("A1", n, p, g, a1, Hnz / I)
                # Theorem A(3)
                vB = bessel_sum(M)
                assert v <= vB + TOL and vB < min(1, math.sqrt(2 / (math.pi * N * p * q))) + TOL
                # Lemma tilt at lambda = 1/(nv)
                assert n * v >= 1 - TOL
                if n * v > 1 + TOL:
                    lam = 1 / (n * v)
                    tilt = (1 - v) * log2(1 / lam) + (n * v - 1) * log2(1 / (1 - lam)) + v * log2(1 / (n * lam * v * peff))
                else:
                    tilt = v * log2(1 / peff)
                assert Hnz / I <= tilt + TOL, ("tilt", n, p, g)
                # Lemma dich
                k, k0, _ = dich_k(n, g, p)
                assert k + 1 in J and piJ >= binom_pmf(N, k, p) - TOL
                vub = min(1.0, vB)
                if k != k0:
                    d = n * p - k if k < k0 else k + 1 - n * p
                    assert d >= 1 - TOL and v <= 1 / d + TOL, ("dich", n, p, g, k, v, d)
                    vub = min(vub, 1 / d)
                # eq:master with sigma = v_* = min(v_ub, 1/(n p_lb)), p_lb = 4pq pi_p(k)
                plb = 4 * p * q * binom_pmf(N, k, p)
                sg = min(vub, 1 / (n * plb))
                E = H / I - log2(1 / (p * q))
                Er = Hraw / I
                tail = (1 - sg) * log2(n * sg * p * q) + sg * log2(1 / binom_pmf(N, k, p))
                assert E < (1 + sg) / L2 - 2 * sg + tail + TOL, ("master E", n, p, g)
                assert Er < 1 / L2 + 2 - 2 * sg + tail + TOL, ("master Eraw", n, p, g)
                # Lemma poi (gibbs) at alpha = 7/2 and 4
                for a in (3.5, 4.0):
                    rhs = log2(1 / (p * q)) + (Cint(a) + v * math.log(Qpoi(M, a) / (4 * v * piJ))) / L2
                    assert Hnz / I <= rhs + 1e-7, ("gibbs", a, n, p, g)
                # regime count
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
                checks += 9
    print(f"n <= {NMAX}, p in {PS}: {checks} checks passed")
    print("regime counts 1..5:", regimes[1:])
    for key, val in worst.items():
        print(f"  largest {key:8s} {val[0]:>9}  at n={val[1]}, p={val[2]}, g={val[3]}")


main()
