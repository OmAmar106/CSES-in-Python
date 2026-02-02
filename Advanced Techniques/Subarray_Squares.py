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
#Bucket Sorted list: bsortl, Segment Tree(lp,selfop): SegmentTree, bootstrap: bootstrap, Trie: tries
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, Segment Tree(lp): SegmentOther
#Graph1(dnc,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, nummat: matrix, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

def Knuths(n,C = lambda x: x):
    # conditions
    # For all a<=b<=c<=d:
    #	 cost(a,c)+cost(b,d)<=cost(a,d)+cost(b,c)
    # opt[l][r-1] <= opt[l][r] <= opt[l+1][r]
    dp = [[0]*n for _ in range(n)]
    opt = [[0]*n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
        opt[i][i] = i
    for i in range(n-2,-1,-1):
        for j in range(i+1,n):
            cost = C(i, j)
            mini = float('inf')
            L = opt[i][j-1]
            R = min(j-1, opt[i+1][j])
            for k in range(L, R+1):
                val = dp[i][k] + dp[k+1][j] + cost
                if val<=mini:
                    mini = val
                    opt[i][j] = k
            dp[i][j] = mini
    return dp[0][n-1]

def DnC(prev, dp, cost):
    # conditions
    # opt[i][j] <= opt[i][j+1]

    # for i in range(1, K + 1):
    #	 DnC(dp_prev, dp_cur, cost, n)
    #	 dp_prev, dp_cur = dp_cur, dp_prev
    def solve(l, r, opt_l, opt_r):
        if l > r:
            return

        mid = (l + r) // 2
        best_k = -1
        best_val = float('inf')
        end = max(0,min(mid - 1, opt_r))
        for k in range(opt_l, end + 1):
            val = prev[k] + cost(k, mid)
            if val < best_val:
                best_val = val
                best_k = k
        dp[mid] = best_val
        solve(l, mid - 1, opt_l, best_k)
        solve(mid + 1, r, best_k, opt_r)
    solve(0,len(dp)-1,0,len(dp)-1)

def solve():
    n,k = LII()
    L = LII()
    
    pref = [0]
    for i in L:
        pref.append(pref[-1]+i)

    prev = [float('inf')]*(n+1)
    prev[0] = 0
    dp = [0]*(n+1)

    for i in range(k):
        DnC(prev,dp,lambda x,y:(pref[y]-pref[x])**2)
        prev,dp = dp,prev
    
    print(prev[-1])

    #L1 = LII()
    #st = SI()
solve()