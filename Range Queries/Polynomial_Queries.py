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
#2-D BIT: 2DBIT, MonoDeque: mono, nummat: matrix
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

class SegmentTree:
    def __init__(self, L):
        self.n = len(L)
        self.N = 1<<(self.n-1).bit_length()
        size = self.N << 1
        self.tree = [0] * size
        self.lazy = [0] * size     
        self.lazy1 = [0] * size 
        for i in range(self.n):
            self.tree[self.N + i] = L[i]
        for i in range(self.N - 1, 0, -1):
            self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]

    def _apply(self, p, length, base, slope):
        self.tree[p] += base * length + slope * length * (length + 1) // 2
        if p < self.N:
            self.lazy[p] += base
            self.lazy1[p] += slope

    def _push(self, p):
        s = self.N.bit_length()
        for h in range(s, 0, -1):
            i = p >> h
            if self.lazy[i] != 0 or self.lazy1[i] != 0:
                length = 1 << (h - 1)
                self._apply(i << 1, length, self.lazy[i], self.lazy1[i])
                offset = self.lazy[i] + self.lazy1[i] * length
                self._apply(i << 1 | 1, length, offset, self.lazy1[i])
                self.lazy[i] = self.lazy1[i] = 0

    def _rebuild(self, p):
        while p > 1:
            p >>= 1
            self.tree[p] = self.tree[p << 1] + self.tree[p << 1 | 1]
            if self.lazy[p] or self.lazy1[p]:
                length = (self.N >> (p.bit_length() - 1))
                self.tree[p] += self.lazy[p] * length + self.lazy1[p] * length * (length + 1) // 2

    def update(self, l, r):
        l0, r0 = l + self.N, r + self.N
        self._push(l0)
        self._push(r0)
        length = 1
        base_left = 0
        base_right = r - l 
        l1, r1 = l0, r0
        while l1 <= r1:
            if l1 & 1:
                self._apply(l1, length, base_left, 1)
                base_left += length
                l1 += 1
            if not r1 & 1:
                self._apply(r1, length, base_right - length + 1, 1)
                base_right -= length
                r1 -= 1
            l1 >>= 1
            r1 >>= 1
            length <<= 1
        self._rebuild(l0)
        self._rebuild(r0)

    def query(self, l, r):
        l += self.N
        r += self.N
        self._push(l)
        self._push(r)
        res = 0
        while l <= r:
            if l & 1:
                res += self.tree[l]
                l += 1
            if not r & 1:
                res += self.tree[r]
                r -= 1
            l >>= 1
            r >>= 1
        return res

def solve():
    n,q = LII()
    L = LII()

    seg = SegmentTree(L)

    for i in range(q):
        t,a,b = LII()
        if t==1:
            seg.update(a-1,b-1)
            # for i in range(42,51):
            #     print(seg.query(i,i),end=' ')
            # print()
        else:
            print(seg.query(a-1,b-1))
    #L1 = LII()
    #st = SI()
solve()