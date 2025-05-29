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
#Graph1(dnc,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphother
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

class MergeSortTree:
    @staticmethod
    def merge(a, b):
        # Change this function depending upon needs
        L = []
        i = 0
        j = 0
        while i<len(a) and j<len(b):
            if a[i]<b[j]:
                L.append(a[i])
                i += 1
            else:
                L.append(b[j])
                j += 1
        while i<len(a):
            L.append(a[i])
            i += 1
        while j<len(b):
            L.append(b[j])
            j += 1
        return L
    @staticmethod
    def op(a,b):
        return a+b
    @staticmethod
    def func(L,l,r):
        return bisect_right(L,r)-bisect_left(L,l)
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)
    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = [data[i]]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.merge(self.tree[i * 2], self.tree[i * 2 + 1])
    def update(self, pos, value):
        # Update the value at the leaf node
        pos += self.n
        # For updating
        self.tree[pos] = value
        # self.tree[pos] += value
        # If you want to add rather than update
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.merge(self.tree[2 * pos], self.tree[2 * pos + 1])
    def query(self, left, right, l, r):
        # Query the maximum value in the range [left, right)
        left += self.n
        right += self.n
        # Change the initializer depending upon the self.func
        max_val = 0
        ##
        while left < right:
            if left % 2:
                max_val = self.op(max_val,self.func(self.tree[left],l,r))
                left += 1
            if right % 2:
                right -= 1
                max_val = self.op(max_val,self.func(self.tree[right],l,r))
            left //= 2
            right //= 2
        return max_val

def solve():
    # n = II()
    n,q = LII()
    seg = MergeSortTree(LII())
    ans = []
    for _ in range(q):
        a,b,c,d = LII()
        ans.append(str(seg.query(a-1,b,c,d)))
    print('\n'.join(ans))
    #L1 = LII()
    #st = SI()
solve()