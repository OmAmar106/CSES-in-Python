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
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class RollbackDSU:
    def __init__(self,n):
        self.p=list(range(n)); self.sz=[1]*n; self.st=[]
        self.size = n
    def find(self,x):
        while x!=self.p[x]: x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b: self.st.append((-1,-1,-1)); return
        self.size -= 1
        if self.sz[a]>self.sz[b]: a,b=b,a
        self.st.append((a,b,self.sz[b]))
        self.p[a]=b; self.sz[b]+=self.sz[a]
    def rollback(self):
        a,b,s=self.st.pop()
        if a==-1: return
        self.size += 1
        self.p[a]=a; self.sz[b]=s

class ParPersistentDSU:
    # Partially Persistent DSU with no branching
    def __init__(self,n):
        self.parent = list(range(n))
        self.size = [1]*n
        self.time = [float('inf')]*n
    def find(self,node,version):
        # returns root at given version
        while not (self.parent[node]==node or self.time[node]>version):
            node = self.parent[node]
        return node
    def union(self,a,b,time):
        # merges a and b
        a = self.find(a,time)
        b = self.find(b,time)
        if a==b:
            return False
        if self.size[a]>self.size[b]:
            a,b = b,a
        self.parent[a] = b
        self.time[a] = time
        self.size[b] += self.size[a]
        return True
    def isconnected(self,a,b,time):
        return self.find(a,time)==self.find(b,time)

class OfflineDynamicConnectivity:
    def __init__(self,n):
        self.n = n
        self.active = {}
        self.ranges = []
        self.max_time = 0
    def add_edge(self,u,v,t):
        if u>v:
            u,v = v,u
        self.active[(u,v)] = t
        self.max_time = max(self.max_time,t)
    def remove_edge(self,u,v,t):
        if u>v:
            u,v = v,u
        start = self.active.pop((u,v))
        self.ranges.append((start,t,u,v))
        self.max_time = max(self.max_time,t)
    def run(self):
        dsu = RollbackDSU(self.n)
        for (u,v), start in self.active.items():
            self.ranges.append((start,self.max_time+1,u,v))
        T = self.max_time + 1
        base = T
        tree = [[] for _ in range(2 * base)]
        for l,r,u,v in self.ranges:
            l += base
            r += base
            while l<r:
                if l&1:
                    tree[l].append((u,v))
                    l += 1
                if r&1:
                    r -= 1
                    tree[r].append((u,v))
                l >>= 1
                r >>= 1
        ans = [None] * T
        stack = [(1,0)]
        while stack:
            node,state = stack.pop()
            if state==0:
                for u,v in tree[node]:
                    dsu.union(u,v)
                stack.append((node,1))
                if node<base:
                    stack.append((node*2+1,0))
                    stack.append((node*2,0))
                else:
                    t = node-base
                    if t<T:
                        ans[t] = dsu.size
                        # implement a size function
                        # size-- on union
            else:
                for _ in tree[node]:
                    dsu.rollback()
        return ans
    
def solve():
    n,m,k = LII()
    
    odc = OfflineDynamicConnectivity(n)

    for _ in range(m):
        u,v = LII_1()
        odc.add_edge(u,v,0)

    for _ in range(k):
        t,u,v = LII_1()
        if t==0:
            odc.add_edge(u,v,_+1)
        else:
            odc.remove_edge(u,v,_+1)
    # print(ranges)
    ans = odc.run()
    print(*ans)

    #L1 = LII()
    #st = SI()
solve()