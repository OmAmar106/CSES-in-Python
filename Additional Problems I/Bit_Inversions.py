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
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, DynamicConnectivity: odc
#Graph1(axtree,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class Node:
    def __init__(self,left0,right0,left1,right1,maxi,is1=False,is0=False):
        self.left0 = left0
        self.right0 = right0
        self.left1 = left1
        self.right1 = right1
        self.maxi = maxi
        self.is1 = is1
        self.is0 = is0
    def func(left,right):
        if right is None:
            if left is None:
                return None
            return left
        elif left is None:
            return right
        return Node(max(left.left0,(left.left0+right.left0) if left.is0 else 0),max(right.right0,(right.right0+left.right0) if right.is0 else 0),max(left.left1,(left.left1+right.left1) if left.is1 else 0),max(right.right1,(right.right1+left.right1) if right.is1 else 0),max(left.maxi,right.maxi,left.right0+right.left0,left.right1+right.left1),left.is1&right.is1,left.is0&right.is0)
    def add(self):
        self.is0 = not self.is0
        self.is1 = not self.is1
        self.right1 = 1-self.right1
        self.left1 = 1-self.left1
        self.left0 = 1-self.left0
        self.right0 = 1-self.right0
    def set(self,val):
        self.val = val
    def __repr__(self):
        return str(self.val)

class SegmentTree:
    #Remember to change the func content as well as the initializer to display the content
    @staticmethod
    def func(a, b):
        return Node.func(a,b)
    def __init__(self, data):
        self.n = 1<<(len(data).bit_length())
        self.tree = [None] * (self.n<<1)
        self.build(data)
    def build(self, data):
        for i in range(len(data)):
            if data[i]:
                self.tree[self.n + i] = Node(0,0,1,1,1,1,0)
            else:
                self.tree[self.n+i] = Node(1,1,0,0,1,0,1)
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i<<1], self.tree[(i<<1) + 1])
    def __getitem__(self,pos):
        if isinstance(pos,slice):
            return self.query(pos.start,pos.stop)
        return self.tree[pos+self.n]
    def up(self,pos):
        pos += self.n
        self.tree[pos].add()
        while pos > 1:
            pos >>= 1
            self.tree[pos] = self.func(self.tree[pos<<1], self.tree[(pos<<1) + 1])
    def query(self, left, right):
        left += self.n
        right += self.n
        res_left = None
        res_right = None
        while left<right:
            if left&1:
                res_left = Node.func(res_left, self.tree[left])
                left += 1
            if right&1:
                right -= 1
                res_right = Node.func(self.tree[right], res_right)
            left >>= 1
            right >>= 1
        return Node.func(res_left, res_right)
    def __repr__(self):
        values = [str(self.query(i, i + 1)) for i in range(self.n)]
        return f"Seg[{', '.join(values)}]"

def solve():
    #L1 = LII()
    st = list(map(int,SI()))
    n = II()
    L = LII_1()
    seg = SegmentTree(st)
    ans = []
    for i in range(len(L)):
        seg.up(L[i])
        ans.append(seg.tree[1].maxi)
    print(*ans)
solve()