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
#2-D BIT: 2DBIT
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

INF = float('inf')
class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def __call__(self, x):
        return self.m * x + self.b
class ConvexHull:
    def __init__(self, n=500000):
        # put n equal to max value of ai , bi , you may need to do coordinate compression in case it is upto 10**9
        # works for value which are not increasing as well
        self.n = n
        self.seg = [Line(0, INF)] * (4 * n)
        self.lo = [0] * (4 * n)
        self.hi = [0] * (4 * n)
        self.build(1,1,n)
    def build(self, i, l, r):
        stack = [(i, l, r)]
        while stack:
            idx, left, right = stack.pop()
            self.lo[idx] = left
            self.hi[idx] = right
            self.seg[idx] = Line(0, INF)
            if left == right:
                continue
            mid = (left + right) // 2
            stack.append((2 * idx + 1, mid + 1, right))
            stack.append((2 * idx, left, mid))
    def insert(self,L):
        pos = 1
        while True:
            l, r = self.lo[pos], self.hi[pos]
            if l == r:
                if L(l) < self.seg[pos](l):
                    self.seg[pos] = L
                break
            m = (l + r) // 2
            if self.seg[pos].m < L.m:
                self.seg[pos], L = L, self.seg[pos]
            if self.seg[pos](m) > L(m):
                self.seg[pos], L = L, self.seg[pos]
                pos = 2*pos
            else:
                pos = 2*pos+1
    def query(self,x):
        i = 1
        res = self.seg[i](x)
        pos = i
        while True:
            l, r = self.lo[pos], self.hi[pos]
            if l == r:
                return min(res, self.seg[pos](x))
            m = (l + r) // 2
            if x < m:
                res = min(res, self.seg[pos](x))
                pos = 2 * pos
            else:
                res = min(res, self.seg[pos](x))
                pos = (2 * pos + 1)

class CHT:
    # Works only for increasing value of input
    def __init__(self, tp):
        self.t = tp
        self.ptr = 0
        self.v = []
    def bad(self, l1, l2, l3):
        a = (l3.c - l1.c) * (l1.m - l2.m)
        b = (l2.c - l1.c) * (l1.m - l3.m)
        if self.t in [1, 4]:
            return a <= b
        return a >= b
    def add(self, line):
        self.v.append(line)
        while len(self.v) >= 3 and self.bad(self.v[-3], self.v[-2], self.v[-1]):
            self.v.pop(-2)
    def val(self, ind, x):
        return self.v[ind].m * x + self.v[ind].c
    def query(self, x):
        l, r = 0, len(self.v) - 1
        ans = 0
        while l <= r:
            mid1 = l + (r - l) // 3
            mid2 = r - (r - l) // 3
            if self.t & 1:
                if self.val(mid1, x) <= self.val(mid2, x):
                    r = mid2 - 1
                    ans = self.val(mid1, x)
                else:
                    l = mid1 + 1
                    ans = self.val(mid2, x)
            else:
                if self.val(mid1, x) >= self.val(mid2, x):
                    r = mid2 - 1
                    ans = self.val(mid1, x)
                else:
                    l = mid1 + 1
                    ans = self.val(mid2, x)
        return ans

    def query2(self, x):
        if not self.v:
            return 0
        if self.ptr >= len(self.v):
            self.ptr = len(self.v) - 1
        while self.ptr < len(self.v) - 1:
            if self.t & 1:
                if self.val(self.ptr, x) > self.val(self.ptr + 1, x):
                    self.ptr += 1
                else:
                    break
            else:
                if self.val(self.ptr, x) < self.val(self.ptr + 1, x):
                    self.ptr += 1
                else:
                    break
        return self.val(self.ptr, x)
    

def f(line, x):
    return line[0] * x + line[1]
 
class LiChao:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.m = (lo + hi) // 2
        self.line = None
        self.left = None
        self.right = None
    def add_line(self, new_line):
        l, r, m = self.lo, self.hi, self.m
        if self.line is None:
            self.line = new_line
            return
        if f(new_line, m) > f(self.line, m):
            self.line, new_line = new_line, self.line
        if l == r:
            return
        if f(new_line, l) > f(self.line, l):
            if self.left is None:
                self.left = LiChao(l, m)
            self.left.add_line(new_line)
        elif f(new_line, r) > f(self.line, r):
            if self.right is None:
                self.right = LiChao(m + 1, r)
            self.right.add_line(new_line)
    def query(self, x):
        res = f(self.line, x) if self.line is not None else -10**18
        if self.lo == self.hi:
            return res
        if x <= self.m and self.left is not None:
            res = max(res, self.left.query(x))
        elif x > self.m and self.right is not None:
            res = max(res, self.right.query(x))
        return res

def solve():
    n,q = LII()
    a = LII()
    b = LII()
    cx = LiChao(-10**6-1,10**6+1)
    cx.add_line((q,0))
    for i in range(len(a)):
        ans = cx.query(-a[i])
        cx.add_line((b[i],ans))
    print(-ans)
    #L1 = LII()
    #st = SI()
solve()