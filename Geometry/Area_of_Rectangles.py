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
#Segment Tree(Node): SegmentNode, mcmf: mcmf, pref2d: pref2d
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class Node:
    def __init__(self,val,count=1):
        self.val = val
        self.count = count
    def func(left,right):
        if right is None:
            if left is None:
                return None
            return left
        elif left is None:
            return right
        if left.val<right.val:return Node(left.val,left.count)
        elif left.val>right.val:return Node(right.val,right.count)
        else:return Node(left.val,left.count+right.count)
    def add(self,val):
        self.val += val
class SegmentTree:
    class _RangeProxy:
        def __init__(self, seg, sl):
            self.seg, self.sl = seg, sl
        def __iadd__(self, v):
            self.seg.range_update(self.sl.start, self.sl.stop-1, v, True)
            return self
        def __repr__(self):
            return str(self.seg.range_query(self.sl.start, self.sl.stop-1))
    def __getitem__(self, k):
        if isinstance(k, slice):
            return SegmentTree._RangeProxy(self, k)
        return self.range_query(k, k)
    def __setitem__(self, k, v):
        if isinstance(v, SegmentTree._RangeProxy):
            return
        if isinstance(k, slice):
            self.range_update(k.start, k.stop-1, v, False)
        else:
            self.range_update(k, k, v, False)
    def __init__(self, data,func=Node.func):
        self.func = func
        self.n = len(data)
        self.size = 1 << (self.n - 1).bit_length()
        self.tree = [None for i in range (2 * self.size)]
        self._size = [0] * (2 * self.size)
        self._size[self.size:] = [1] * self.size
        for i in range(self.size - 1, 0, -1):
            self._size[i] = self._size[i << 1] + self._size[i << 1 | 1]
        self.lazy_add = 0
        self.lazy_add = [0] * self.size
        for i in range(self.n):
            self.tree[self.size + i] = Node(data[i])
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i << 1], self.tree[i << 1 | 1])
    def _apply_set(self, pos, value):
        self.tree[pos].set(value)
        if pos < self.size:
            self.lazy_add[pos] = 0
    def _apply_add(self, pos, value):
        self.tree[pos].add(value)
        if pos < self.size:
            self.lazy_add[pos] += value
    def _build(self, pos):
        while pos > 1:
            pos >>= 1
            self.tree[pos] = self.func(self.tree[pos << 1], self.tree[pos << 1 | 1])
            if self.lazy_add[pos] != 0:
                self.tree[pos].add(self.lazy_add[pos])
    def _push(self, pos):
        for shift in range(self.size.bit_length() - 1, 0, -1):
            i = pos >> shift
            add_val = self.lazy_add[i]
            if add_val != 0:
                self._apply_add(i << 1, add_val)
                self._apply_add(i << 1 | 1, add_val)
                self.lazy_add[i] = 0
    def range_update(self, left, right, value,flag=True):
        # Range Update in [L,R] if flag, then add
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
    def range_query(self, left, right):
        # Range Query in [L,R]
        l = left + self.size
        r = right + self.size
        self._push(l)
        self._push(r)
        lefty = None
        righty = None
        while l <= r:
            if l & 1: lefty = Node.func(lefty,self.tree[l]); l += 1
            if not r & 1: righty = Node.func(self.tree[r],righty); r -= 1
            l >>= 1; r >>= 1
        return Node.func(lefty,righty)

def solve():
    n = II()
    L = []
    t = -float('inf')
    t1 = float('inf')
    for i in range(n):
        x1,y1,x2,y2 = LII()
        t = max(t,y2)
        t1 = min(t1,y1)
        L.append((x1,y1,y2,1))
        L.append((x2,y1,y2,-1))
    L.sort()
    MAXI = t-t1+1
    # print(L)
    seg = SegmentTree([0]*MAXI)
    prev = L[0][0]
    i = 0
    ans = 0
    while i<len(L):
        f = L[i][0]
        ans += (f-prev)*(MAXI-seg.range_query(0,MAXI-1).count)
        # print(list(seg)[1:10],MAXI-seg.range_query(0,MAXI-1).count,prev,f)
        # print(f,ans,list(seg)[1:10])
        while i<len(L) and L[i][0]==f:
            seg[L[i][1]-t1:L[i][2]-t1] += L[i][3]
            i += 1
        prev = f

    print(ans)

    #L1 = LII()
    #st = SI()
solve()