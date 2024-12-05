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
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq
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

def bfs(graph, start=0):
    used = [False] * len(graph)
    used[start] = True
    q = [start]
    for v in q:
        for w in graph[v]:
            if not used[w]:
                used[w] = True
                q.append(w)

def dfs(graph, start=0):
    n = len(graph)
    dp = [0] * n
    visited, finished = [False] * n, [False] * n
    stack = [start]
    while stack:
        start = stack[-1]
        if not visited[start]:
            visited[start] = True
            for child in graph[start]:
                if not visited[child]:
                    stack.append(child)
        else:
            stack.pop()
            dp[start] += 1
            for child in graph[start]:
                if finished[child]:
                    dp[start] += dp[child]
            finished[start] = True

    return visited, dp

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        acopy = a
        while a != self.parent[a]:
            a = self.parent[a]
        while acopy != a:
            self.parent[acopy], acopy = a, self.parent[acopy]
        return a

    def merge(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def kruskal(n, U, V, W):
    union = UnionFind(n)
    cost, merge_cnt = 0, 0
    mst_u, mst_v = [], []
    order = sorted(range(len(W)), key=lambda x: W[x])
    for i in range(len(W)):
        u, v = U[order[i]], V[order[i]]
        find_u, find_v = union.find(u), union.find(v)
        if find_u != find_v:
            cost += W[order[i]]
            merge_cnt += 1
            union.parent[find_v] = find_u
            mst_u.append(u), mst_v.append(v)

    return cost, mst_u, mst_v, n == 1 + merge_cnt

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    L3 = []
    L4 = []
    L5 = []
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        L4.append(L1[1]-1)
        L3.append(L1[0]-1)
        L5.append(L1[2])
    cost,_,_,flag = kruskal(L[0],L3,L4,L5)
    if not flag:
        print("IMPOSSIBLE")
        return
    print(cost)
    #st = sys.stdin.readline().strip()
solve()