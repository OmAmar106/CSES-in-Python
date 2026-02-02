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
#2-D BIT: 2DBIT, MonoDeque: mono, nummat: matrix, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

def extras():
    getcontext().prec = 50
    sys.setrecursionlimit(10**6)
    sys.set_int_max_str_digits(10**5)
# extras()

def interactive():
    import builtins
    # print(globals())
    globals()['print'] = lambda *args, **kwargs: builtins.print(*args, flush=True, **kwargs)
# interactive()

def GI(n,m=None,sub=-1,dirs=False,weight=False):
    if m==None:
        m = n-1
    d = [[] for i in range(n)]
    if not weight:
        for i in range(m):
            u,v = LII_C(lambda x:int(x)+sub)
            d[u].append(v)
            if not dirs:
                d[v].append(u)
    else:
        for i in range(m):
            u,v,w = LII()
            d[u+sub].append((v+sub,w))
            if not dirs:
                d[v+sub].append((u+sub,w))
    return d

ordalp = lambda s : ord(s)-65 if s.isupper() else ord(s)-97
alp = lambda x : chr(97+x)
yes = lambda : print("Yes")
no = lambda : print("No")
yn = lambda flag : print("Yes" if flag else "No")
printf = lambda x : print(-1 if x==float('inf') else x)
lalp = 'abcdefghijklmnopqrstuvwxyz'
ualp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
dirs = ((1,0),(0,1),(-1,0),(0,-1))
dirs8 = ((1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1),(1,1),(-1,-1))
ldir = {'D':(1,0),'U':(-1,0),'R':(0,1),'L':(0,-1)}

def find_bridges(adj):
    # returns all bridges
    bridges = []
    n = len(adj)
    timer = 0
    visited = [False]*n
    tin = [-1]*n
    low = [-1]*n
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, -1, 0, False)]
        visited[start] = True
        tin[start] = low[start] = timer
        timer += 1
        while stack:
            v, parent, idx, backtrack = stack.pop()
            if backtrack:
                to = adj[v][idx]
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges.append((v,to))
                continue
            if idx < len(adj[v]):
                to = adj[v][idx]
                stack.append((v, parent, idx + 1, False))
                if to == parent:
                    continue
                if visited[to]:
                    low[v] = min(low[v], tin[to])
                else:
                    visited[to] = True
                    tin[to] = low[to] = timer
                    timer += 1
                    stack.append((v, parent, idx, True))
                    stack.append((to, v, 0, False))
    return bridges

def bridges_on_path(adj):
    # returns all bridges on path from 1 to n
    n = len(adj)
    timer = 0
    visited = [False]*n
    tin = [-1]*n
    low = [-1]*n
    bridges = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, -1, 0, False)]
        visited[start] = True
        tin[start] = low[start] = timer
        timer += 1
        while stack:
            v, parent, idx, backtrack = stack.pop()
            if backtrack:
                to = adj[v][idx]
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges.append((v, to))
                continue
            if idx < len(adj[v]):
                to = adj[v][idx]
                stack.append((v, parent, idx+1, False))
                if to == parent:
                    continue
                if visited[to]:
                    low[v] = min(low[v], tin[to])
                else:
                    visited[to] = True
                    tin[to] = low[to] = timer
                    timer += 1
                    stack.append((v, parent, idx, True))
                    stack.append((to, v, 0, False))
    comp_id = [-1]*n
    comp = 0
    bridge_set = set(bridges)
    for i in range(n):
        if comp_id[i] != -1:
            continue
        stack = [i]
        comp_id[i] = comp
        while stack:
            v = stack.pop()
            for to in adj[v]:
                if comp_id[to] == -1 and (v, to) not in bridge_set and (to, v) not in bridge_set:
                    comp_id[to] = comp
                    stack.append(to)
        comp += 1

    tree = [[] for _ in range(comp)]
    for u, v in bridges:
        cu, cv = comp_id[u], comp_id[v]
        tree[cu].append((cv, (u, v)))
        tree[cv].append((cu, (u, v)))

    c1, cN = comp_id[0], comp_id[n-1]
    if c1 == cN:
        return []
    parent = {c1: None}
    edge_used = {c1: None}
    stack = [c1]
    while stack:
        u = stack.pop()
        for v, e in tree[u]:
            if v not in parent:
                parent[v] = u
                edge_used[v] = e
                stack.append(v)
    path_bridges = []
    cur = cN
    while cur != c1:
        path_bridges.append(edge_used[cur])
        cur = parent[cur]
    return [(u,v) for u, v in path_bridges]

def lowlink(edge):
    n = len(edge)
    parent = [-1] * n
    visited = [False] * n
    for s in range(n):
        if not visited[s]:
            que = [s]
            while que:
                now = que.pop()
                if visited[now]: continue
                visited[now] = True
                for nxt in edge[now]:
                    if not visited[nxt]:
                        parent[nxt] = now
                        que.append(nxt)
    order = [-1] * n
    low = [-1] * n
    is_articulation = [False] * n
    articulation = []
    bridge = []
    def dfs(s):
        idx = 0
        cnt = 0
        que = [~s,s]
        while que:
            now = que.pop()
            if now >= 0:
                order[now] = low[now] = idx
                idx += 1
                for nxt in edge[now]:
                    if parent[nxt] == now:
                        que.append(~nxt)
                        que.append(nxt)
                    elif parent[now] != nxt and order[nxt] != -1:
                        low[now] = min(low[now], order[nxt])
            else:
                now = ~now
                par = parent[now]
                if par == s: cnt += 1
                if now == s:
                    is_articulation[now] |= (cnt >= 2)
                    if is_articulation[now]:
                        articulation.append(now)
                    return
                if is_articulation[now]:
                    articulation.append(now)
                if now != parent[par]:
                    low[par] = min(low[par], low[now])
                is_articulation[par] |= (par != s) and (order[par] <= low[now])
                if order[par] < low[now]:
                    bridge.append((par, now))
    for i in range(n):
        if parent[i] == -1:
            dfs(i)
    return articulation, bridge

def find_2ecc(edges,d):
    # returns a new graph, in which two nodes are connected
    # if and only if they are part of same cycle.
    _,bridges = lowlink(d)
    newd = [[] for i in range(len(d))]
    bridges = set((w(i[0]),w(i[1])) for i in bridges)
    for u,v in edges:
        if (w(u),w(v)) not in bridges and (w(v),w(u)) not in bridges:
            newd[u].append(v)
            newd[v].append(u)
    return newd

def solve():
    n,m = LII()
    d = GI(n,m)
    L = find_bridges(d)
    print(len(L))
    for x,y in L:
        print(x+1,y+1)

    #L1 = LII()
    #st = SI()
solve()