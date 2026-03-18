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
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, DynamicConnectivity: odc
#Treap: treap, MaxMatching(Kuhn,Hopcroft): graphmatch
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class HLPP:
    # O(V²E), still sometime unexpectedly faster than Dinic
    def __init__(self,n):
        self.n=n
        self.g=[[] for _ in range(n)]
    def add_edge(self,u,v,c):
        self.g[u].append([v,c,len(self.g[v])])
        self.g[v].append([u,0,len(self.g[u])-1])
    def maxflow(self,s,t):
        n,g=self.n,self.g
        h=[0]*n; ex=[0]*n
        seen=[0]*n
        h[s]=n
        for v,c,rev in g[s]:
            if c:
                g[v][rev][1]+=c
                ex[v]+=c
        q=deque([i for i in range(n) if i!=s and i!=t and ex[i]>0])
        def push(u,e):
            v,c,rev=e
            d=min(ex[u],c)
            if h[u]==h[v]+1 and d>0:
                e[1]-=d; g[v][rev][1]+=d
                ex[u]-=d; ex[v]+=d
                if ex[v]==d and v!=s and v!=t:
                    q.append(v)
                return True
            return False
        def relabel(u):
            h[u]=min(h[v]+1 for v,c,_ in g[u] if c>0)
        while q:
            u=q[0]
            while ex[u]:
                if seen[u]<len(g[u]):
                    if not push(u,g[u][seen[u]]):
                        seen[u]+=1
                else:
                    relabel(u)
                    seen[u]=0
            q.popleft()
        return ex[t]

class Dinic:
    # O(V²E)
    # Unit capacities/bipartite/DAG: ~O(E√V)
    def __init__(self, n):
        self.lvl = [0] * n
        self.ptr = [0] * n
        self.q = [0] * n
        self.adj = [[] for _ in range(n)]
    def add_edge(self, a, b, c, rcap=0):
        self.adj[a].append([b, len(self.adj[b]), c, 0])
        self.adj[b].append([a, len(self.adj[a]) - 1, rcap, 0])
    def dfs(self, v, t, f):
        if v == t or not f:
            return f
        for i in range(self.ptr[v], len(self.adj[v])):
            e = self.adj[v][i]
            if self.lvl[e[0]] == self.lvl[v] + 1:
                p = self.dfs(e[0], t, min(f, e[2] - e[3]))
                if p:
                    self.adj[v][i][3] += p
                    self.adj[e[0]][e[1]][3] -= p
                    return p
            self.ptr[v] += 1
        return 0
    def calc(self, s, t):
        INF = float('inf')
        flow, self.q[0] = 0, s
        for l in range(31):  # l = 30 maybe faster for random data
            while True:
                self.lvl, self.ptr = [0] * len(self.q), [0] * len(self.q)
                qi, qe, self.lvl[s] = 0, 1, 1
                while qi < qe and not self.lvl[t]:
                    v = self.q[qi]
                    qi += 1
                    for e in self.adj[v]:
                        if not self.lvl[e[0]] and (e[2] - e[3]) >> (30 - l):
                            self.q[qe] = e[0]
                            qe += 1
                            self.lvl[e[0]] = self.lvl[v] + 1
                p = self.dfs(s, t, INF)
                while p:
                    flow += p
                    p = self.dfs(s, t, INF)
                if not self.lvl[t]:
                    break
        return flow
    def get_paths(self,s,t):
        # returns that paths taken for the maxflow
        paths=[]
        while True:
            stack=[(s,[],float('inf'),set([s]))]
            found=False
            while stack:
                v,path,f,vis=stack.pop()
                if v==t:
                    pushed=f
                    res_path=path
                    found=True
                    break
                for i,e in enumerate(self.adj[v]):
                    to,rev,cap,flow=e
                    if flow>0 and to not in vis:
                        stack.append((to,path+[(v,to)],min(f,flow),vis|{to}))
            if not found:
                break
            f=pushed
            for u,w in res_path:
                for e in self.adj[u]:
                    if e[0]==w and e[3]>0:
                        e[3]-=f
                        self.adj[w][e[1]][3]+=f
                        break
            paths.append((pushed,res_path))
        return paths

def solve():
    n,m = LII()
    dnc = Dinic(n)

    for _ in range(m):
        u,v = LII_1()
        dnc.add_edge(u,v,1)

    def f(y):
        for u,v in y:
            print(u+1,end=' ')
        print(y[-1][-1]+1)

    print(dnc.calc(0,n-1))
    ans = dnc.get_paths(0,n-1)
    for x,y in ans:
        print(len(y)+1)
        (f(y))
    #L1 = LII()
    #st = SI()
solve()