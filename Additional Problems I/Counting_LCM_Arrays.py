import sys,math,cmath,random,os
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations,combinations
from io import BytesIO, IOBase
from decimal import Decimal,getcontext

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
MOD = 10**9 + 7
RANDOM = random.randrange(1,2**62)
def gcd(a,b):
    while b:
        a,b = b,a%b
    return a
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
MI = lambda x : x(map(int, sys.stdin.readline().split()))
SI = lambda : sys.stdin.readline().strip()
SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
##

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare, SparseTable: SparseTable
#Bucket Sorted list: bsortl, bootstrap: bootstrap, Trie: tries, Segment Tree(lp): SegmentOther, Treap: treap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, bwt: bwt, DynamicConnectivity: odc
#Graph1(axtree,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, Suffix/KMPAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode, mcmf: mcmf, pref2d: pref2d, RecSegTree: SegmentTreeRec
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class Matrix:
    MOD = 10**9+7
    @staticmethod
    def transpose(mat):
        return [list(c) for c in zip(*mat)]
    def __init__(self,M):
        self.M=M
    def minmul(self,o):
        L,B=self.M,o.M
        r=len(L); c=len(B[0])
        return Matrix([[min(L[i][k]+B[k][j] for k in range(len(B)))%self.MOD for j in range(c)] for i in range(r)])
    def __matmul__(self,o):
        L,B=self.M,o.M
        r=len(L); c=len(B[0])
        return Matrix([[sum(L[i][k]*B[k][j] for k in range(len(B)))%self.MOD for j in range(c)] for i in range(r)])
    def __pow__(self,p):
        M=self; n=len(M.M)
        R=Matrix([[i==j for j in range(n)] for i in range(n)])
        while p:
            R = R@M if p&1 else R
            M = M@M
            p//=2
        return R
    @staticmethod
    def gauss(A,mod=10**9+7):
        m,n=len(A),len(A[0])-1; r=0; L=[-1]*n
        for c in range(n):
            for i in range(r,m):
                if A[i][c]: A[r],A[i]=A[i],A[r]
                break
            else:
                continue
            k=pow(A[r][c],-1,mod)
            for j in range(c,n+1):
                A[r][j]=A[r][j]*k%mod
            for i in range(m):
                if i!=r and A[i][c]:
                    f=A[i][c]
                    for j in range(c,n+1):
                        A[i][j]=(A[i][j]-f*A[r][j])%mod
            L[c]=r
            r+=1
        if any(A[i][n] for i in range(r,m)):
            return None
        return [A[L[i]][n] if L[i]!=-1 else 0 for i in range(n)]
    def __str__(self):
        return '\n'.join(' '.join(map(str,row)) for row in self.M)
    def __repr__(self):
        return f"Matrix({self.M})"
    def __iter__(self):
        return iter(self.M)

def memodict(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__

def pollard_rho(n):
    # returns a random factor of n
    if n&1==0:return 2
    if n%3==0:return 3
    s = ((n-1)&(1-n)).bit_length() - 1
    d = n>>s
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        p = pow(a, d, n)
        if p == 1 or p == n - 1 or a % n == 0:continue
        for _ in range(s):
            prev = p
            p = (p * p) % n
            if p == 1:return gcd(prev - 1, n)
            if p == n - 1:break
        else:
            for i in range(2, n):
                x, y = i, (i * i + 1) % n
                f = gcd(abs(x - y), n)
                while f == 1:
                    x, y = (x * x + 1) % n, (y * y + 1) % n
                    y = (y * y + 1) % n
                    f = gcd(abs(x - y), n)
                if f != n:return f
    return n

@memodict
def prime_factors_large(n):
    # returns prime factor in n^(1/4) but is probablistic
    if n <= 1:return Counter()
    f = pollard_rho(n)
    return Counter([n]) if f == n else prime_factors_large(f) + prime_factors_large(n // f)

def discrete_log(a, b, mod):
    # returns smalest x such that pow(a,x,mod) = b
    n = int(mod**0.5) + 1
    tiny_step, e = {}, 1
    for j in range(1, n + 1):
        e = e * a % mod
        if e == b:
            return j
        tiny_step[b * e % mod] = j
    factor = e
    for i in range(2, n + 2):
        e = e * factor % mod
        if e in tiny_step:
            j = tiny_step[e]
            return n * i - j if pow(a, n * i - j, mod) == b else None
    return None

def extended_gcd(a, b):
    # returns gcd(a, b), s, r s.t. a * s + b * r == gcd(a, b)
    s, old_s = 0, 1
    r, old_r = b, a
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    return old_r, old_s, (old_r - old_s * a) // b if b else 0

def composite_crt(b, m):
    # returns x s.t. x = b[i] (mod m[i]) for all i
    x, m_prod = 0, 1
    for bi, mi in zip(b, m):
        g, s, _ = extended_gcd(m_prod, mi)
        if ((bi - x) % mi) % g:
            return None
        x += m_prod * (s * ((bi - x) % mi) // g)
        m_prod = (m_prod * mi) // gcd(m_prod, mi)
    return x % m_prod

def phi(n):
    ph = [i if i & 1 else i // 2 for i in range(n + 1)]
    for i in range(3,n+1,2):
        if ph[i]==i:
            for j in range(i,n+1,i):
                ph[j] = (ph[j]//i)*(i-1)
    return ph

def solve():
    n,k = LII()

    # dp[i][0] = dp[i][1]*(val)
    # dp[i][1] = dp[i][0]+dp[i][1]
    ans = 1
    d = prime_factors_large(k)
    for j in d:
        mat = Matrix([[0,d[j]],[1,1]])
        ans *= sum(sum(x) for x in ((mat**(n+1))@(Matrix([[0,1]]))).M)
        ans %= MOD

    print(ans)
    #L1 = LII()
    #st = SI()
for _ in range(II()):
    #if os.environ.get('LOCAL'):print('________________'+'Test Case : '+str(_+1)+'________________\n')
    solve()