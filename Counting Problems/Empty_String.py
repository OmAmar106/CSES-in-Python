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
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

# class Factorial:
#     def __init__(self, N, mod):
#         assert(N<mod)
#         N += 1
#         self.mod = mod
#         self.f = [1 for _ in range(N)]
#         self.g = [1 for _ in range(N)]
#         for i in range(1, N):
#             self.f[i] = self.f[i - 1] * i % self.mod
#         self.g[-1] = pow(self.f[-1], mod - 2, mod)
#         for i in range(N - 2, -1, -1):
#             self.g[i] = self.g[i + 1] * (i + 1) % self.mod
#     def derangements(self,n):
#         # Number of permutations of n objects where no object appears in its original position
#         return int(self.fac(n)/math.e+0.5)
#     def der(self, n):
#         if n==0: return 1
#         if n==1: return 0
#         a, b = 1, 0  # der(0), der(1)
#         for i in range(2,n+1):
#             a,b = b,((i-1)*(a+b)) % self.mod
#         return b
#     def stirling_2(self,n,k):
#         # Number of ways to partition n elements into k non-empty subsets
#         return sum(((-1)**(k - j)) * math.comb(k, j) * (j**n) for j in range(k + 1)) // math.factorial(k)
#     def stirling_2_mod(self,n,k):
#         return (sum(((-1)**(k - j))*self.combi(k, j)*pow(j,n,self.mod) for j in range(k + 1)) * self.fac_inv(k))%self.mod
#     def partition(self,n,k):
#         # Ways to partition n into k or fewer parts of size 1 or greater
#         # add dp[(n,k)] and memoize it if using it.
#         if n<0 or (k<1 and n>0):return 0
#         if n==0:return 1
#         return 1 if n==1 else self.partition(n,k-1)+self.partition(n-k,k)
#     def lucas_nCk(self,n,k):
#         # Lucas's theorem for finding ((nCk)%p) for large n,k
#         # in log p(n)
#         ans = 1
#         p = self.mod
#         while n or k:
#             x = n%p
#             y = k%p
#             ans *= self.combi(x,y)
#             ans %= p
#             n //= p
#             k //= p
#         return ans
#     def catalan(self, n):
#         return (self.combi(2 * n, n) - self.combi(2 * n, n - 1)) % self.mod
#     def fac(self, n):
#         return self.f[n]
#     def fac_inv(self, n):
#         return self.g[n]
#     def combi(self, n, m):
#         if m == 0: return 1
#         if n < m or m < 0 or n < 0: return 0
#         return self.f[n] * self.g[m] % self.mod * self.g[n - m] % self.mod
#     def permu(self, n, m):
#         if n < m or m < 0 or n < 0: return 0
#         return self.f[n] * self.g[n - m] % self.mod
#     def inv(self, n):
#         return self.f[n-1] * self.g[n] % self.mod

# Lfac = Factorial(501,MOD)

combi = [[0]*501 for i in range(501)]
combi[0][0] = 1

for i in range(501):
    for j in range(501):
        if not (i==0 and j==0):
            combi[i][j] = (combi[i-1][j]+combi[i-1][j-1])%MOD

# print(combi[1][1])
dp = [[0]*(501) for i in range(501)]

def solve():
    #L1 = LII()
    st = SI()
    n = len(st)

    for i in range(n-1):
        if st[i]==st[i+1]:
            dp[i][i+1] = 1
        dp[i+1][i] = 1
            # dp[i][i] = 1

    for i in range(3,n,2):
        for j in range(n-i):
            for k in range(j+1,j+i+1,2):
                if st[j]==st[k]:
                    dp[j][j+i] = (dp[j][j+i]+dp[j+1][k-1]*(dp[k+1][j+i] if k+1<n else 1)*combi[(i+1)//2][(k-j+1)//2])%MOD

    # print(dp[0][1],dp[2][5],dp[3][4])
    print(dp[0][n-1])            

solve()