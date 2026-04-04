import sys,math,cmath,random,os
# from heapq import heappush,heappop
# from bisect import bisect_right,bisect_left
# from collections import Counter,deque,defaultdict
# from itertools import permutations,combinations
from io import BytesIO, IOBase
# from decimal import Decimal,getcontext

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
# MOD = 10**9 + 7
# RANDOM = random.randrange(1,2**62)
# def gcd(a,b):
#     while b:
#         a,b = b,a%b
#     return a
# def lcm(a,b):
#     return a//gcd(a,b)*b
# def w(x):
#     return x ^ RANDOM
II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
# MI = lambda x : x(map(int, sys.stdin.readline().split()))
# SI = lambda : sys.stdin.readline().strip()
# SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
# LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
# LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
# MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
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
#Segment Tree(Node): SegmentNode, mcmf: mcmf
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class LiChao:
    __slots__ = ("lo","hi","mid","m","b","left","right")
    def __init__(self,lo,hi):
        self.lo = lo
        self.hi = hi
        self.mid = (lo+hi)>>1
        self.m = None
        self.b = None
        self.left = None
        self.right = None
    def add_line(self,m,b):
        node = self
        while True:
            lo,hi,mid = node.lo,node.hi,node.mid
            if node.m is None:
                node.m,node.b = m,b
                return
            cur_m,cur_b = node.m,node.b
            if m*mid+b>cur_m*mid+cur_b:
                node.m,m = m,cur_m
                node.b,b = b,cur_b
                cur_m,cur_b = node.m,node.b
            if lo==hi:
                return
            if m*lo+b>cur_m*lo+cur_b:
                if node.left is None:
                    node.left = LiChao(lo,mid)
                node = node.left
            elif m*hi+b>cur_m*hi +cur_b:
                if node.right is None:
                    node.right = LiChao(mid+1,hi)
                node = node.right
            else:
                return
    def query(self,x):
        node = self
        res = -float('inf')
        while node:
            if node.m is not None:
                v = node.m*x+node.b
                if v>res:
                    res = v
            if node.lo==node.hi:
                break
            if x<=node.mid:
                node = node.left
            else:
                node = node.right
        return res


        # self.size = 1 << (self.n - 1).bit_length()
size = 1<<17
tree = [None for i in range (2*size)]
def range_update(left, right, value):
    # Range Update in [L,R] if flag, then add
    l = left + size
    r = right + size
    while l <= r:
        # print('Y',l,r)
        if l & 1: 
            if not tree[l]:
                tree[l] = LiChao(0,1<<17)
            tree[l].add_line(*value)
            l += 1
        if not r & 1: 
            if not tree[r]:
                tree[r] = LiChao(0,1<<17)
            tree[r].add_line(*value)
            r -= 1
        l >>= 1; r >>= 1
def range_query(pos):
    # Range Query in [L,R]
    x = pos+1
    pos += size
    ans = tree[pos].query(x) if tree[pos] else -float('inf')
    while pos>1:
        pos >>= 1
        ans = max(ans,tree[pos].query(x) if tree[pos] else -float('inf'))
    return ans

def solve():
    for _ in range(II()):
        L = LII()
        if L[0]==1:
            _,a,b,l,r = L
            range_update(l-1,r-1,(a,b))
        else:
            ans = (range_query(L[1]-1))
            print(ans if ans!=-float('inf') else "NO")
    #L1 = LII()
    #st = SI()
solve()