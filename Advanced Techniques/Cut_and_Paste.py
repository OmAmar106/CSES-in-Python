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
#Graph1(axtree,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, Graphflow(hlpp,dnc): graphflow, DynamicConnectivity: odc
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class Node:
    __slots__ = ("val", "priority", "left", "right", "size")
    def __init__(self, val):
        self.val = val
        self.priority = random.random()
        self.left = None
        self.right = None
        self.size = 1
class Treap:
    def __init__(self, iterable=None):
        self.root = None
        if iterable:
            for x in iterable:
                self.root = Treap._merge(self.root, Node(x))
    @staticmethod
    def _size(node):
        return node.size if node else 0
    @staticmethod
    def _update(node):
        if node:
            node.size = 1 + Treap._size(node.left) + Treap._size(node.right)
    @staticmethod
    def _merge(left, right):
        if not left or not right:
            return left or right

        if left.priority > right.priority:
            left.right = Treap._merge(left.right, right)
            Treap._update(left)
            return left
        else:
            right.left = Treap._merge(left, right.left)
            Treap._update(right)
            return right
    def _kth(node, k):
        if not node:
            return None

        left_size = Treap._size(node.left)

        if k < left_size:
            return Treap._kth(node.left, k)
        elif k == left_size:
            return node.val
        else:
            return Treap._kth(node.right, k - left_size - 1)
    def kth(self, k):
        if k < 0 or k >= Treap._size(self.root):
            return None
        return Treap._kth(self.root, k)
    @staticmethod
    def merge(t1, t2):
        t = Treap()
        t.root = Treap._merge(t1.root, t2.root)
        return t
    @staticmethod
    def _split(node, k):
        if not node:
            return (None, None)
        left_size = Treap._size(node.left)
        if k <= left_size:
            left, node.left = Treap._split(node.left, k)
            Treap._update(node)
            return (left, node)
        else:
            node.right, right = Treap._split(node.right, k - left_size - 1)
            Treap._update(node)
            return (node, right)
    def insert(self, pos, val):
        new_node = Node(val)
        left, right = Treap._split(self.root, pos)
        merged = Treap._merge(left, new_node)
        self.root = Treap._merge(merged, right)
    def split(self, k):
        left_root, right_root = Treap._split(self.root, k)
        t1 = Treap()
        t2 = Treap()
        t1.root = left_root
        t2.root = right_root
        return t1, t2
    def __iter__(self):
        stack = []
        node = self.root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left

            node = stack.pop()
            yield node.val
            node = node.right
    
def solve():
    n,q = LII()
    st = SI()

    trp = Treap([i for i in range(n)])
    
    for _ in range(q):
        l,r = LII_1()
        left,mid = trp.split(l)
        mid,right = mid.split((r-l+1))
        trp.root = Treap._merge(Treap._merge(left.root, right.root), mid.root)

    print(''.join(st[i] for i in trp))

    #L1 = LII()
    #st = SI()
solve()