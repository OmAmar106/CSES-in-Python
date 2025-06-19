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
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

class BIT2D:
    def __init__(self, arr):
        self.n = len(arr)
        self.m = len(arr[0]) if self.n > 0 else 0
        # self.bit = [row[:] for row in arr]
        self.bit = arr # assuming that arr is not used after this
        for i in range(self.n):
            for j in range(self.m):
                ni = i | (i + 1)
                if ni < self.n:
                    self.bit[ni][j] += self.bit[i][j]
        for i in range(self.n):
            for j in range(self.m):
                nj = j | (j + 1)
                if nj < self.m:
                    self.bit[i][nj] += self.bit[i][j]
    def add(self, x, y, delta):
        # 0-based in log n * log m
        i = x
        while i < self.n:
            j = y
            while j < self.m:
                self.bit[i][j] += delta
                j |= j + 1
            i |= i + 1
    def sum(self, x, y):
        # sum from 0,0 to x,y inclusive in log n * log m
        if not (0<=x<self.n) or not (0<=y<self.m):
            return 0
        res = 0
        i = x
        while i >= 0:
            j = y
            while j >= 0:
                res += self.bit[i][j]
                j = (j & (j + 1)) - 1
            i = (i & (i + 1)) - 1
        return res
    def query(self, x1, y1, x2, y2):
        # sum of L[x1:x2+1][y1:y2+1]
        return (self.sum(x2,y2)-self.sum(x1-1,y2)-self.sum(x2,y1-1)+(self.sum(x1-1,y1-1)))

def solve():
    n,q = LII()
    L = []
    d = {'*':1,'.':0}
    for i in range(n):
        L.append(list(map(lambda x:d[x],SI())))
    
    bit = BIT2D(L)
    d = {1:-1,0:1}
    for _ in range(q):
        L = LII()
        if L[0]==2:
            print(bit.query(L[1]-1,L[2]-1,L[3]-1,L[4]-1))
        else:
            bit.add(L[1]-1, L[2]-1, d[bit.query(L[1]-1, L[2]-1, L[1]-1, L[2]-1)])
    #L1 = LII()
    #st = SI()
solve()