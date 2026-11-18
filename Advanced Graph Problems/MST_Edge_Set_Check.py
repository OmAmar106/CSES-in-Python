import sys,math,cmath,random,os
# from heapq import heappush,heappop
# from bisect import bisect_right,bisect_left
from collections import defaultdict
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
# II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
# MI = lambda x : x(map(int, sys.stdin.readline().split()))
SI = lambda : sys.stdin.readline()
# SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
# LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
# MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
##

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare/numfast, SparseTable: SparseTable
#Bucket sortedlist: bsortl, bootstrap: bootstrap, Trie: tries, Segment Tree(lp): SegmentOther, Treap: treap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, DynamicConnectivity: odc
#Graph1(axtree,bl): graphadv, Graph2(khn,sat,scc): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, Suffix/KMPAutomaton: sautomaton, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, MaxFlow(MinCut,Dinic,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode, mcmf: mcmf, pref2d: pref2d, RecSegTree: SegmentTreeRec, AVL sortedlist: asortl
#bwt: bwt, linalg: linalg, LCT: lct, Rollback/Par DSU: rbdsu
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class RollbackDSU:
    def __init__(self,n):
        self.p=list(range(n)); self.sz=[1]*n; self.st=[]
    def find(self,x):
        while x!=self.p[x]: x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b: self.st.append((-1,-1,-1)); return False
        if self.sz[a]>self.sz[b]: a,b=b,a
        self.st.append((a,b,self.sz[b]))
        self.p[a]=b; self.sz[b]+=self.sz[a]
        return True
    def rollback(self):
        a,b,s=self.st.pop()
        if a==-1: return
        self.p[a]=a; self.sz[b]=s

import gc
gc.disable()

n = 100000
parent = list(range(n))
visited = [False]*n
size = [1] * n
def find(a):
    acopy = a
    while a != parent[a]:
        a = parent[a]
    while acopy!=a:
        parent[acopy], acopy = a, parent[acopy]
    return a
def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

MASK = (1<<20)-1

m = 200000
d3 = [-1]*m
ans = [True]*m

def solve():
    n,m,q = LII()

    edges = []


    for i in range(m):
        u,v,w = LII_1()
        edges.append((w<<60)+(u<<40)+(v<<20)+i)

    edges.sort()

    i = 0
    while i<len(edges):
        f = (edges[i]>>60)
        j = i
        while i<len(edges) and (edges[i]>>60)==f:
            # edges[i] = (f+1,ds.find(edges[i][1]),ds.find(edges[i][2]),edges[i][3])
            d3[edges[i]&MASK] = ((find((edges[i]>>40)&MASK)<<20)+find((edges[i]>>20)&MASK))
            i += 1
        while j<i:
            # _,u,v,ind = edges[j]
            u = (edges[j]>>40)&MASK
            v = (edges[j]>>20)&MASK
            ind = (edges[j])&MASK
            if u==v:
                ans[ind] = False
            else:
                union(u,v)
            j += 1

    fans = []

    rb = RollbackDSU(n)

    for _ in range(q):
        SI()
        L = LII_1()

        f = True

        for i in L:
            if not ans[i]:
                f = False
            if not rb.union(d3[i]>>20,(d3[i]&MASK)):
                f = False

        for i in L:
            rb.rollback()

        fans.append("NO" if not f else "YES")

    print('\n'.join(fans))

    #L1 = LII()
    #st = SI()
solve()