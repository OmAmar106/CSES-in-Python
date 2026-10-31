import sys,math,cmath,random,os
# from heapq import heappush,heappop
# from bisect import bisect_right,bisect_left
# from collections import Counter,deque,defaultdict
# from itertools import permutations,combinations
from io import BytesIO, IOBase
# from decimal import Decimal,getcontext

BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

# functions #
# MOD = 998244353
# MOD = 10**9 + 7
# RANDOM = random.randrange(1,2**62)
# def gcd(a,b):
#     while b:
#         a,b = b,a%b
#     return a
# def lcm(a,b):
#     return a//gcd(a,b)*b
# def w(x):
#     return x ^ RANDOM
II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
# MI = lambda x : x(map(int, sys.stdin.readline().split()))
# SI = lambda : sys.stdin.readline().strip()
# SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
# LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
# LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
# MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
##

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare, SparseTable: SparseTable
#Bucket sortedlist: bsortl, bootstrap: bootstrap, Trie: tries, Segment Tree(lp): SegmentOther, Treap: treap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, DynamicConnectivity: odc
#Graph1(axtree,bl): graphadv, Graph2(khn,sat,scc): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, Suffix/KMPAutomaton: sautomaton, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode, mcmf: mcmf, pref2d: pref2d, RecSegTree: SegmentTreeRec, AVL sortedlist: asortl
#bwt: bwt, linalg: linalg, LCT: lct
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

# def solve_quadratic(a, b, c):
#     if a == 0:return None
#     D = b*b - 4*a*c
#     if D < 0:return None
#     s = math.isqrt(D)
#     if s*s != D:return None
#     den = 2*a
#     if den == 0:return None
#     x1, r1 = divmod(-b + s, den)
#     x2, r2 = divmod(-b - s, den)
#     if r1!=0 or r2!=0:return None
#     return (x1, x2)

# def solve_cubic(a, b, c, d):
#     if a == 0:return None
#     roots = []
#     for x in range(-math.isqrt(abs(d)) - 1, math.isqrt(abs(d)) + 2):
#         if a*x*x*x + b*x*x + c*x + d == 0:
#             roots.append(x)
#     return tuple(roots) if roots else None

# def compute_rank(A):
#     EPS = 1e-9
#     A = [row[:] for row in A]
#     n = len(A)
#     m = len(A[0])
#     rank = 0
#     row_selected = [False] * n
#     for i in range(m):
#         j = 0
#         while j < n:
#             if not row_selected[j] and abs(A[j][i]) > EPS:
#                 break
#             j += 1
#         if j != n:
#             rank += 1
#             row_selected[j] = True
#             for p in range(i + 1, m):
#                 A[j][p] /= A[j][i]
#             for k in range(n):
#                 if k != j and abs(A[k][i]) > EPS:
#                     for p in range(i + 1, m):
#                         A[k][p] -= A[j][p] * A[k][i]
#     return rank


# def simplex_bland(c, A, b, frac=False):
#     # maximize ci.xi
#     # given ai.xi <= bi and xi >= 0
#     if not frac:
#         n = len(c)
#         m = len(b)
#         T = [A[i] + [0]*i + [1] + [0]*(m-i-1) + [b[i]] for i in range(m)]
#         T.append([-ci for ci in c] + [0]*m + [0])
#         N = list(range(n)) # non-basic
#         B = list(range(n, n+m)) # basic
#         def pivot(r, s):
#             # Make coeffecient of pivot row 1 and Eliminate column s in all other rows
#             T[r] = [v / T[r][s] for v in T[r]]
#             for i in range(len(T)):
#                 if i != r:
#                     fac = T[i][s]
#                     T[i] = [T[i][j] - fac * T[r][j] for j in range(len(T[0]))]
#         while True:
#             s = None
#             for j in range(len(T[0]) - 1):
#                 if T[-1][j] < 0:
#                     s = j
#                     break
#             if s is None:
#                 break
#             # leaving row
#             r = None
#             for i in range(m):
#                 if T[i][s] > 1e-12:
#                     ratio = T[i][-1] / T[i][s]
#                     if r is None or ratio < T[r][-1] / T[r][s]:
#                         r = i
#             if r is None:
#                 return None
#             pivot(r, s)
#             B[r], N[s] = N[s], B[r]
#         x = [0] * (n + m)
#         for i in range(m):
#             x[B[i]] = T[i][-1]
#         return x[:n], T[-1][-1]
#     else:
#         from fractions import Fraction
#         n = len(c)
#         m = len(b)
#         c = [Fraction(ci) for ci in c]
#         A = [[Fraction(aij) for aij in row] for row in A]
#         b = [Fraction(bi) for bi in b]
#         T = [A[i] + [Fraction(1 if i == j else 0) for j in range(m)] + [b[i]]
#              for i in range(m)]
#         T.append([-ci for ci in c] + [Fraction(0)] * (m + 1))
#         N = list(range(n))
#         B = list(range(n, n+m))
#         def pivot(r, s):
#             piv = T[r][s]
#             T[r] = [v / piv for v in T[r]]
#             for i in range(m + 1):
#                 if i != r:
#                     fac = T[i][s]
#                     T[i] = [T[i][j] - fac * T[r][j] for j in range(len(T[0]))]
#         while True:
#             s = None
#             for j in range(len(T[0]) - 1):
#                 if T[-1][j] < 0:
#                     s = j
#                     break
#             if s is None:
#                 break
#             r = None
#             for i in range(m):
#                 if T[i][s] > 0:
#                     ratio = T[i][-1] / T[i][s]
#                     if r is None or ratio < T[r][-1] / T[r][s]:
#                         r = i
#             if r is None:
#                 return None
#             pivot(r, s)
#             B[r], N[s] = N[s], B[r]
#         x = [Fraction(0)] * (n + m)
#         for i in range(m):
#             x[B[i]] = T[i][-1]
#         return x[:n], T[-1][-1]

# def s_lr(l,r,pow=1):
#     if l>r:return 0
#     if pow==1:return (r*(r+1)//2) - ((l-1)*l//2)
#     elif pow==2:return (r*(r+1)*(2*r+1)//6) - ((l-1)*l*(2*l-1)//6)
#     elif pow==3:return (r*(r+1)//2)**2 - ((l-1)*l//2)**2
#     else:return None

# class XorBasis:
#     def __init__(self,A):
#         self.reduced_base = {}
#         for i in A:
#             self.add(i)
#     def max(self):
#         base = self.base; reduced_base = self.reduced_base
#         how = self.how
#         x = 0;tmp = 0;ans=[]
#         for j in sorted(reduced_base, reverse=True):
#             if not x & (1 << j):
#                 x ^= reduced_base[j]
#                 tmp ^= how[j]
#                 ans.append(base[j])
#         # xor of these elements return the max element
#         return ans
#     def make(self,k):
#         # returns True if can make x with these elements
#         reduced_base = self.reduced_base
#         for j in range(k.bit_length()-1,-1,-1):
#             if ((1<<j)&k):
#                 if j in reduced_base:
#                     k ^= reduced_base[j]
#                 else:
#                     return False
#         return k==0
#     def add(self,a):
#         # adds a to the basis
#         reduced_base = self.reduced_base
#         while a:
#             b = a.bit_length() - 1
#             if b in reduced_base:
#                 a ^= reduced_base[b]
#             else:
#                 reduced_base[b] = a
#                 break

def solve():
    n,k = LII()

    L = [0]*30
    ans = []
    for a in LII():
        while a:
            b = a.bit_length() - 1
            if L[b]:
                a ^= L[b]
            else:
                L[b] = a
                ans.append(a)
                break
    z = n-len(ans)
    # L = list(L.values())
    # L.sort()
    # L.reverse()
    L = sorted(ans,reverse=True)

    fans = []

    st = [(0,0)]
    while st:
        val,ind = st.pop()
        if ind==len(L):
            t = (1<<z)
            fans += [val]*min(t,k-len(fans))
            if len(fans)==k:
                break
            continue
        if val&(1<<(L[ind].bit_length()-1)):
            st.append((val,ind+1))
            st.append((val^(L[ind]),ind+1))
        else:
            st.append((val^L[ind],ind+1))
            st.append((val,ind+1))

    print(*fans)

    #L1 = LII()
    #st = SI()
solve()