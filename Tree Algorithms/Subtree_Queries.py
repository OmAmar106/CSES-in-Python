import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

def bellman_ford(n, edges, start):
    dist = [float("inf")] * n
    pred = [None] * n
    dist[start] = 0
    for _ in range(n):
        for u, v, d in edges:
            if dist[u] + d < dist[v]:
                dist[v] = dist[u] + d
                pred[v] = u
    # for u, v, d in edges:
    #	 if dist[u] + d < dist[v]:
    #		 return -1
    # This returns -1 , if there is a negative cycle

class binary_lift:
    def __init__(self, graph, data=(), f=min, root=0):
        n = len(graph)
        parent = [-1] * (n + 1)
        depth = self.depth = [-1] * n
        bfs = [root]
        depth[root] = 0
        for node in bfs:
            for nei in graph[node]:
                if depth[nei] == -1:
                    parent[nei] = node
                    depth[nei] = depth[node] + 1
                    bfs.append(nei)

        data = self.data = [data]
        parent = self.parent = [parent]
        self.f = f

        for _ in range(max(depth).bit_length()):
            old_data = data[-1]
            old_parent = parent[-1]
            data.append([f(val, old_data[p]) for val,p in zip(old_data, old_parent)])
            parent.append([old_parent[p] for p in old_parent])

    def lca(self, a, b):
        depth = self.depth
        parent = self.parent
        if depth[a] < depth[b]:
            a,b = b,a
        d = depth[a] - depth[b]
        for i in range(d.bit_length()):
            if (d >> i) & 1:
                a = parent[i][a]
        for i in range(depth[a].bit_length())[::-1]:
            if parent[i][a] != parent[i][b]:
                a = parent[i][a]
                b = parent[i][b]
        if a != b:
            return parent[0][a]
        else:
            return a
    def distance(self, a, b):
        return self.depth[a] + self.depth[b] - 2 * self.depth[self.lca(a,b)]
    def kth_ancestor(self, a, k):
        parent = self.parent
        if self.depth[a] < k:
            return -1
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                a = parent[i][a]
        return a
    def __call__(self, a, b):
        depth = self.depth
        parent = self.parent
        data = self.data
        f = self.f
        c = self.lca(a, b)
        val = data[0][c]
        for x,d in (a, depth[a] - depth[c]), (b, depth[b] - depth[c]):
            for i in range(d.bit_length()):
                if (d >> i) & 1:
                    val = f(val, data[i][x])
                    x = parent[i][x]
        return val

def floyd_warshall(n, edges):
    dist = [[0 if i == j else float("inf") for i in range(n)] for j in range(n)]
    pred = [[None] * n for _ in range(n)]

    for u, v, d in edges:
        dist[u][v] = d
        pred[u][v] = u

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]
    # Sanity Check
    # for u, v, d in edges:
    #	 if dist[u] + d < dist[v]:
    #		 return None

    return dist, pred

def dijkstra(graph, start ,n):
    dist, parents = [float("inf")] * n, [-1] * n
    dist[start] = 0
    queue = [(0, start)]
    while queue:
        path_len, v = heappop(queue)
        if path_len == dist[v]:
            for w, edge_len in graph[v]:
                if edge_len + path_len < dist[w]:
                    dist[w], parents[w] = edge_len + path_len, v
                    heappush(queue, (edge_len + path_len, w))
    return dist, parents

def toposort(graph):
    res, found = [], [0] * len(graph)
    stack = list(range(len(graph)))
    while stack:
        node = stack.pop()
        if node < 0:
            res.append(~node)
        elif not found[node]:
            found[node] = 1
            stack.append(~node)
            stack += graph[node]
    for node in res:
        if any(found[nei] for nei in graph[node]):
            return None
        found[node] = 0
    return res[::-1]

def kahn(graph):
    n = len(graph)
    indeg, idx = [0] * n, [0] * n
    for i in range(n):
        for e in graph[i]:
            indeg[e] += 1
    q, res = [], []
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)
    nr = 0
    while q:
        res.append(q.pop())
        idx[res[-1]], nr = nr, nr + 1
        for e in graph[res[-1]]:
            indeg[e] -= 1
            if indeg[e] == 0:
                q.append(e)
    return res, idx, nr == n

def rec(cur,color):
    # If asking for SCC, rather than d, use the reversed graph
    # Also the traversel should be in reverse of topological order
    visited = [],ans = [],d = {} # remove this
    visited[cur] = True
    ans[cur] = color
    for i in d[cur]:
        if visited[i]:
            continue
        ans1 = (yield rec(i,color))
    yield -1

def euler_path(d):
    start = [1]
    ans = []
    while start:
        cur = start[-1]
        if len(d[cur])==0:
            ans.append(cur)
            start.pop()
            continue
        k1 = d[cur].pop()
        d[k1].remove(cur)
        start.append(k1)
    return ans

INF = float("inf")
class Dinic:
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

class BIT:
    #Faster than segment tree so use if possbile
    def __init__(self, x):
        """transform list into BIT"""
        self.bit = x
        for i in range(len(x)):
            j = i | (i + 1)
            if j < len(x):
                x[j] += x[i]
    def update(self, idx, x):
        """updates bit[idx] += x"""
        #basically after that number greater greater than x will be added
        while idx < len(self.bit):
            self.bit[idx] += x
            idx |= idx + 1
    def query(self, end):
        """calc sum(bit[:end])"""
        #gives sum of element before end
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x
    def findkth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(len(self.bit).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(self.bit) and k >= self.bit[right_idx]:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1

from types import GeneratorType

def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

# @bootstrap
# put this just on top of recursion function to increase the recursion limit

# rather than return now use yield and when function being called inside itself, use yield before the function name
# example usage:
# @bootstrap
# def rec1(L,k,cur,count):
# 	if count>=100000:
# 		yield float('INF')
# 	if cur+k+1>=len(L)-1:
# 		yield L[cur]+2
# 	if cur in d:
# 		yield d[cur]
# 	ans = float('INF')
# 	mini = float('INF')
# 	for i in range(k+1,0,-1):
# 		if L[cur+i]<mini:
# 			ans = min(ans,1+L[cur]+(yield rec1(L,k,cur+i,count+1)))
# 			mini = L[cur+i]
# 	d[cur] = ans
# 	yield ans
# the limit of recursion on cf is 10**6


seg = BIT([0 for i in range(200000)])

def dfs(graph,L):
    starttime = [[0,0] for i in range(len(graph))]
    time = 0
    stack = [(0,-1,0)]
    while stack:
        cur, prev, state = stack.pop()
        if state == 0:
            starttime[cur][0] = time
            seg.update(starttime[cur][0], L[cur])
            time += 1
            stack.append((cur, prev, 1))
            for neighbor in graph[w(cur)]:
                if neighbor == prev:
                    continue
                stack.append((neighbor, cur, 0))    
        elif state == 1:
            starttime[cur][1] = time
    return starttime
 
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(n-1):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[w(L1[0]-1)].append(L1[1]-1)
        d[w(L1[1]-1)].append(L1[0]-1)
    arr = (dfs(d,L))
    for i in range(q):
        L1 = list(map(int, sys.stdin.readline().split()))
        if L1[0]==2:
            print(seg.query(arr[L1[1]-1][1])-seg.query(arr[L1[1]-1][0]))
        else:
            seg.update(arr[L1[1]-1][0],L1[2]-L[L1[1]-1])
            L[L1[1]-1] = L1[2]
    #st = sys.stdin.readline().strip()
solve()