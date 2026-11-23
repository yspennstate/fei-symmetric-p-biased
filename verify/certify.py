"""Interval certificates for the constants in the proof of Theorem B.

Every number printed here is an outward-rounded upper bound computed with
mpmath interval arithmetic at 55 decimal digits.  Run:  python certify.py
"""
from mpmath import iv, mp

iv.dps = 55
mp.dps = 55
LN2 = iv.log(2)
E = iv.exp(1)
PI = iv.pi


def up(x):
    return x.b


def log2(x):
    return iv.log(x) / LN2


def bessel_I(nu, x, terms=400):
    """I_nu(x) for x >= 0, nu in {0, 1}: partial sum plus a geometric tail bound."""
    s = iv.mpf(0)
    t = (x / 2) ** nu / iv.factorial(nu)
    for k in range(terms):
        s += t
        t = t * (x / 2) ** 2 / ((k + 1) * (k + 1 + nu))
    r = (x / 2) ** 2 / ((terms + 1) * (terms + 1 + nu))
    assert up(r) < 1
    return s + iv.mpf([0, up(t / (1 - r))])


def v_B(x):
    return iv.exp(-x) * (bessel_I(0, x) + bessel_I(1, x))


def C_int(a):
    return iv.log(1 + a + iv.sqrt(a * (a + 2))) + 1 - iv.log(a)


def Q(M, a, terms=400):
    """Q(M; alpha) of (eq:Qpoi) on an interval M: partial sum plus geometric tail."""
    b = a / E
    s = iv.mpf(0)
    fac = iv.mpf(1)
    for m in range(terms):
        j = m + 1
        fac = fac * j
        s += M ** m / fac * (j / (j + a * M)) ** j * b ** j
    # tail: term_m <= M^m b^(m+1)/(m+1)!, ratio <= M b/(terms+2)
    j = terms + 1
    t = M ** terms / (fac * j) * b ** j
    r = M * b / (terms + 2)
    assert up(r) < 1
    return s + iv.mpf([0, up(t / (1 - r))])


def gmax(C, vd):
    """An upper bound for max over v in (0, vd] of v ln(C/v).

    v ln(C/v) <= C/e always.  When C/e >= vd is certified the function is
    increasing on (0, vd] and the maximum is the endpoint value vd ln(C/vd).
    """
    if (C / E).a >= vd.b:
        return vd * iv.log(C / vd)
    return C / E


KSTAR = 3 * log2(iv.mpf(3)) - 4
CR = 3 * log2(iv.mpf(3)) - 2 - 1 / LN2
out = []


def ceil4(u):
    """u rounded up to four decimals, as a string (an upper bound for u)."""
    k = int(mp.ceil(mp.mpf(u) * 10000))
    return f"{k // 10000}.{k % 10000:04d}"


def show(name, x, bound, where=""):
    u = up(x)
    ok = u < bound
    out.append((name, u, bound, ok))
    print(f"{name:52s} {ceil4(u):>8s}  < {bound}  {'ok' if ok else 'FAIL'}{where}")


print("analytic constants")
print("  K_*  in", mp.nstr(KSTAR.a, 10), mp.nstr(KSTAR.b, 10))
print("  C_R  in", mp.nstr(CR.a, 10), mp.nstr(CR.b, 10))

# k = 0 and k = N
show("k=0      H/I - log2(1/pq) <= 1/ln2 + 2", 1 / LN2 + 2, 4.25)
show("k=0      Hraw/I <= 1/ln2 + 4", 1 / LN2 + 4, 5.75)
show("k=N      H/I - log2(1/pq) <= 1/ln2 + 1", 1 / LN2 + 1, 4.25)
show("k=N      Hraw/I <= 1/ln2 + 3", 1 / LN2 + 3, 5.75)

# left tail: Phi_L(1) and max of Phi_L^raw
show("left     Phi_L(1)", 19 / (6 * LN2) + log2(PI / 8) / 2, 4.25)
show("left     Phi_L^raw(pi/8 e^(-2/3))", 2 + 2 / LN2 + PI / (16 * iv.exp(iv.mpf(2) / 3) * LN2), 5.75)

# right tail
show("right    Phi_R(1)", 1 + 13 / (6 * LN2) + log2(3 * PI / 8) / 2, 4.25)
vR = 3 * PI / 8 * iv.exp(iv.mpf(1) / 3 - 1 + 2 * (1 - CR) * LN2)
assert 0 < vR.a and vR.b < 1
phiRraw = 2 + (1 + vR / 6) / LN2 + vR / 2 * log2(3 * PI / (8 * vR)) + (1 - vR) * CR + vR
show("right    Phi_R^raw(v_R^*)", phiRraw, 5.75)


def central(a, Mhi, pieces, extra, target):
    """sup over M in [1/3, Mhi] of K_* + extra + [C_int + gmax(C_M, v_B)]/ln 2.

    Start from `pieces` equal pieces; a piece whose enclosure is not below
    `target` is halved, to depth 12.  v_B is decreasing in M (its derivative is
    -e^(-M) I_1(M)/M), so it is evaluated at the left end of each piece.
    Returns (largest certified upper bound, number of pieces used).
    """
    lo = (iv.mpf(1) / 3).a      # a point at or below 1/3, so [1/3, Mhi] is covered
    hi = mp.mpf(Mhi)
    ci = C_int(iv.mpf(a))
    pre = iv.exp(iv.mpf(7) / 6 if extra == 0 else iv.mpf(1) / 6) / 4

    def bound(M1, M2):
        M = iv.mpf([M1, M2])
        C = up(pre * Q(M, iv.mpf(a)) * iv.sqrt(3 * PI * M))
        vd = up(v_B(iv.mpf(M1)))
        g = gmax(iv.mpf(C), iv.mpf(vd))
        return up(KSTAR + extra + (ci + g) / LN2)

    w = (hi - lo) / pieces
    stack = [(lo + i * w, lo + (i + 1) * w, 0) for i in range(pieces)]
    stack[-1] = (stack[-1][0], hi, 0)
    worst, used, at = mp.mpf(0), 0, None
    while stack:
        M1, M2, d = stack.pop()
        b = bound(M1, M2)
        if b >= target and d < 12:
            mid = (M1 + M2) / 2
            stack += [(M1, mid, d + 1), (mid, M2, d + 1)]
            continue
        used += 1
        if b > worst:
            worst, at = b, (M1, M2)
    return worst, used, at


def central_tail(a, M0, extra):
    """M >= M0: Q < (3 sqrt(pi)/8) 2^(1/4) a^(1/4) M^(-3/4), v_B < sqrt(2/(pi M))."""
    M = iv.mpf(M0)
    pre = iv.exp(iv.mpf(7) / 6 if extra == 0 else iv.mpf(1) / 6) / 4
    qb = 3 * iv.sqrt(PI) / 8 * iv.mpf(2) ** 0.25 * iv.mpf(a) ** 0.25
    C = pre * qb * M ** (-0.75) * iv.sqrt(3 * PI * M)   # = c0 M^(-1/4)
    vd = iv.sqrt(2 / (PI * M))                            # = d0 M^(-1/2)
    # C/vd grows like M^(1/4), so C/e >= vd at M0 gives it for all M >= M0;
    # then gmax = vd ln(C/vd) = d0 M^(-1/2) (L + ln(M)/4) with L = ln(c0/d0),
    # decreasing in M once L + ln(M)/4 > 1/2, which is monotone in M.
    assert (C / E).a >= vd.b
    c0 = C * M ** 0.25
    d0 = vd * M ** 0.5
    L = iv.log(c0 / d0)
    assert (L + iv.log(M) / 4).a > 0.5
    return KSTAR + extra + (C_int(iv.mpf(a)) + vd * iv.log(C / vd)) / LN2


import sys
pieces_H, pieces_R = (200, 240) if len(sys.argv) < 2 else (int(sys.argv[1]), int(sys.argv[2]))
def where(at):
    return f"  (worst piece M in [{mp.nstr(at[0], 6)}, {mp.nstr(at[1], 6)}])"


w = central(3.5, 10, pieces_H, 0, 4.25)
show(f"central  H, alpha=7/2, M in [1/3,10], {w[1]} pieces", iv.mpf(w[0]), 4.25, where(w[2]))
show("central  H, alpha=7/2, M >= 10", central_tail(3.5, 10, 0), 4.25)
w = central(4.0, 30, pieces_R, 2, 5.75)
show(f"central  Hraw, alpha=4, M in [1/3,30], {w[1]} pieces", iv.mpf(w[0]), 5.75, where(w[2]))
show("central  Hraw, alpha=4, M >= 30", central_tail(4.0, 30, 2), 5.75)

print("ALL OK" if all(o[3] for o in out) else "SOME BOUND FAILED")
