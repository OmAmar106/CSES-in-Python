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

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare/numfast, SparseTable: SparseTable
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

def derangements(n):
    # Number of permutations of n objects where no object appears in its original position
    return int(math.factorial(n)/math.e+0.5)

def precompute_stirling_1(n,MOD):
    # number of permutations of first n elements that has exactly k cycles
    dp = [[0]*(n+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1,n+1):
        for j in range(1,i+1):
            dp[i][j] = (dp[i-1][j-1]+(i-1)*dp[i-1][j])%MOD
    return dp

def precompute_stirling_2(n,MOD=MOD):
    # stirling[n][k] = No of ways to partition n elements into k non-empty subsets
    stirling = [[0]*(n+1) for i in range(n+1)]
    stirling[0][0] = 1
    for i in range(1,n+1):
        for j in range(1,i+1):
            stirling[i][j] = (j*stirling[i-1][j]+stirling[i-1][j-1])%MOD
    return stirling

def partition(n,k):
    # Ways to partition n into k or fewer parts of size 1 or greater
    # add dp[(n,k)] and memoize it if using it.
    if n<0 or (k<1 and n>0):return 0
    if n==0:return 1
    return 1 if n==1 else partition(n,k-1)+partition(n-k,k)

class Factorial:
    def __init__(self, N, mod):
        assert(N<mod)
        N += 1
        self.mod = mod
        self.f = [1 for _ in range(N)]
        self.g = [1 for _ in range(N)]
        for i in range(1, N):
            self.f[i] = self.f[i - 1] * i % self.mod
        self.g[-1] = pow(self.f[-1], mod - 2, mod)
        for i in range(N - 2, -1, -1):
            self.g[i] = self.g[i + 1] * (i + 1) % self.mod
    def precompute_bell(self,n):
        # number of ways to divide n distinct
        # elements into non empty sets
        # bell[n] = (sum of stirling[n][j])
        bell = [0]*(n+1)
        bell[0] = bell[1] = 1
        for i in range(2,n+1):
            for j in range(i):
                bell[i] += self.combi(i-1,j)*bell[j]
            bell[i] %= MOD
        return bell
    def der(self, n):
        if n==0: return 1
        if n==1: return 0
        a, b = 1, 0  # der(0), der(1)
        for i in range(2,n+1):
            a,b = b,((i-1)*(a+b)) % self.mod
        return b
    def stirling_2_mod(self,n,k):
        # Number of ways to partition n elements into k non-empty subsets
        return (sum(((-1)**(k - j))*self.combi(k, j)*pow(j,n,self.mod) for j in range(k + 1)) * self.fac_inv(k))%self.mod
    def lucas_nCk(self,n,k):
        # Lucas's theorem for finding ((nCk)%p) for large n,k
        # in log p(n)
        ans = 1
        p = self.mod
        while n or k:
            x = n%p
            y = k%p
            ans *= self.combi(x,y)
            ans %= p
            n //= p
            k //= p
        return ans
    def catalan(self,n):
        return (self.combi(2*n,n)-self.combi(2*n,n-1))%self.mod
    def fac(self, n):
        return self.f[n]
    def fac_inv(self, n):
        return self.g[n]
    def combi(self, n, m):
        if m == 0: return 1
        if n < m or m < 0 or n < 0: return 0
        return self.f[n] * self.g[m] % self.mod * self.g[n - m] % self.mod
    def permu(self, n, m):
        if n < m or m < 0 or n < 0: return 0
        return self.f[n] * self.g[n - m] % self.mod
    def inv(self, n):
        return self.f[n-1] * self.g[n] % self.mod
    def beggar(self,n,sumi):
        # number of ways n elements >=0 have sum sumi
        if n==0:return int(sumi==0)
        if sumi<0:return 0
        return self.combi(n+sumi-1,sumi)

Lfac = Factorial(200001,MOD)

val = 5001
A = [[0]*(val+1) for _ in range(val+1)]
A[0][0] = 1

for i in range(1,val+1):
    for j in range(i):
        A[i][j] = ((j+1)*A[i-1][j]+(i-j)*A[i-1][j-1]) % MOD

# A[i][j] = number of ways i elements and exactly j pairs having 
# ai > i

def solve():
    n,a,b = LII()

    z = n-(a+b)
    if z<0:
        print(0)
        return

    # def f(n,a):
    #     # n cards
    #     # a times fp wins
        
    #     def rec(cur,score):
    #         if cur==n+1:
    #             return 1 if score==a else 0
    #         ans = rec(cur+1,score)
    #         ans += (cur-score-1)*rec(cur+1,score+1)
    #         return ans
        
    #     return rec(1,0)

    def f(n,a):
        ans = 0
        for i in range(n-a+1):
            # i elements are same
            if i%2==0:
                ans += Lfac.combi(n,i)*A[n-i][a]
            else:
                ans -= Lfac.combi(n,i)*A[n-i][a]
            ans %= MOD
        return ans
    
    ans = Lfac.combi(n,z)
    print((Lfac.fac(n)*ans*f(n-z,a))%MOD)

    #L1 = LII()
    #st = SI()
for _ in range(II()):
    #if os.environ.get('LOCAL'):print('________________'+'Test Case : '+str(_+1)+'________________\n')
    solve()