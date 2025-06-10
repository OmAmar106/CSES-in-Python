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

class SegmentTree:
    def __init__(self, data, default=0, mode='s'):
        self.func = lambda x,y:x+y
        self.mode = mode
        self._default = default
        self.n = len(data)
        self.size = 1 << (self.n - 1).bit_length()
        self.tree = [default] * (2 * self.size)
        self._size = [0] * (2 * self.size)
        self._size[self.size:] = [1] * self.size
        for i in range(self.size - 1, 0, -1):
            self._size[i] = self._size[i << 1] + self._size[i << 1 | 1]
        self.lazy_add = 0 if self.mode == 's' else 0
        self.lazy_set = None
        self.lazy_add = [0] * self.size
        self.lazy_set = [None] * self.size
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i << 1], self.tree[i << 1 | 1])
    def _apply_set(self, pos, value):
        if self.mode == 's':
            self.tree[pos] = value * self._size[pos]
        else:
            self.tree[pos] = value
        if pos < self.size:
            self.lazy_set[pos] = value
            self.lazy_add[pos] = 0
    def _apply_add(self, pos, value):
        if self.mode == 's':
            self.tree[pos] += value * self._size[pos]
        else:
            self.tree[pos] += value
        if pos < self.size:
            if self.lazy_set[pos] is not None:
                self.lazy_set[pos] += value
            else:
                self.lazy_add[pos] += value
    def _build(self, pos):
        while pos > 1:
            pos >>= 1
            self.tree[pos] = self.func(self.tree[pos << 1], self.tree[pos << 1 | 1])
            if self.lazy_set[pos] is not None:
                if self.mode == 's':
                    self.tree[pos] = self.lazy_set[pos] * self._size[pos]
                else:
                    self.tree[pos] = self.lazy_set[pos]
            if self.lazy_add[pos] != 0:
                if self.mode == 's':
                    self.tree[pos] += self.lazy_add[pos] * self._size[pos]
                else:
                    self.tree[pos] += self.lazy_add[pos]
    def _push(self, pos):
        for shift in range(self.size.bit_length() - 1, 0, -1):
            i = pos >> shift
            set_val = self.lazy_set[i]
            if set_val is not None:
                self._apply_set(i << 1, set_val)
                self._apply_set(i << 1 | 1, set_val)
                self.lazy_set[i] = None
            add_val = self.lazy_add[i]
            if add_val != 0:
                self._apply_add(i << 1, add_val)
                self._apply_add(i << 1 | 1, add_val)
                self.lazy_add[i] = 0
    def range_update(self, left, right, value,flag=True):
        # Range Update in [L,R] if flag, then add
        if flag:
            l = left + self.size
            r = right + self.size
            l0, r0 = l, r
            self._push(l0)
            self._push(r0)
            while l <= r:
                if l & 1: self._apply_add(l, value); l += 1
                if not r & 1: self._apply_add(r, value); r -= 1
                l >>= 1; r >>= 1
            self._build(l0)
            self._build(r0)
        else:
            l = left + self.size
            r = right + self.size
            l0, r0 = l, r
            self._push(l0)
            self._push(r0)
            while l <= r:
                if l & 1: self._apply_set(l, value); l += 1
                if not r & 1: self._apply_set(r, value); r -= 1
                l >>= 1; r >>= 1
            self._build(l0)
            self._build(r0)
    def range_query(self, left, right):
        # Range Query in [L,R]
        l = left + self.size
        r = right + self.size
        self._push(l)
        self._push(r)
        res = self._default
        while l <= r:
            if l & 1: res = self.func(res, self.tree[l]); l += 1
            if not r & 1: res = self.func(res, self.tree[r]); r -= 1
            l >>= 1; r >>= 1
        return res
    def __repr__(self):
        return f"SegmentTree({[self.range_query(i,i) for i in range(self.n)]})"

def solve():
    n,q = LII()
    seg = SegmentTree(LII())
    for i in range(q):
        L = LII_1()
        if L[0]==0:
            seg.range_update(L[1],L[2],L[3]+1)
        elif L[0]==1:
            seg.range_update(L[1],L[2],L[3]+1,flag=False)
        else:
            print(seg.range_query(L[1],L[2]))
        # print(seg)
    #L1 = LII()
    #st = SI()
solve()