import sys,os,random
from heapq import heappush,heappop
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

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


# functions #
MOD1 = random.randint(10**9,10**11)
MOD2 = 10**9 + 7
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
# LII = lambda : list(map(int, sys.stdin.readline().split()))
# MI = lambda x : x(map(int, sys.stdin.readline().split()))
# SI = lambda : sys.stdin.readline().strip()
# SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
# LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
# LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
# MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
##

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lp,selfop): SegmentTree, bootstrap: bootstrap, Trie: tries
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, Segment Tree(lp): SegmentOther
#Graph1(dnc,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

# def extras():
#     getcontext().prec = 50
#     sys.setrecursionlimit(10**6)
#     sys.set_int_max_str_digits(10**5)
# # extras()

# def interactive():
#     import builtins
#     # print(globals())
#     globals()['print'] = lambda *args, **kwargs: builtins.print(*args, flush=True, **kwargs)
# interactive()

# def GI(n,m=None,sub=-1,dirs=False,weight=False):
#     if m==None:
#         m = n-1
#     d = [[] for i in range(n)]
#     if not weight:
#         for i in range(m):
#             u,v = LII_C(lambda x:int(x)+sub)
#             d[u].append(v)
#             if not dirs:
#                 d[v].append(u)
#     else:
#         for i in range(m):
#             u,v,w = LII()
#             d[u+sub].append((v+sub,w))
#             if not dirs:
#                 d[v+sub].append((u+sub,w))
#     return d

# ordalp = lambda s : ord(s)-65 if s.isupper() else ord(s)-97
# alp = lambda x : chr(97+x)
# yes = lambda : print("Yes")
# no = lambda : print("No")
# yn = lambda flag : print("Yes" if flag else "No")
# printf = lambda x : print(-1 if x==float('inf') else x)
# lalp = 'abcdefghijklmnopqrstuvwxyz'
# ualp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# dirs = ((1,0),(0,1),(-1,0),(0,-1))
# dirs8 = ((1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1),(1,1),(-1,-1))
# ldir = {'D':(1,0),'U':(-1,0),'R':(0,1),'L':(0,-1)}

# def bfsf(d,start=0):
#     bfs = [start]
#     visited = [False]*(len(d))
#     visited[start] = True
#     for u in bfs:
#         for j in d[u]:
#             if visited[j]:
#                 continue
#             visited[j] = True
#             bfs.append(j)
#     return bfs

# def dfs(d,start=0):
#     # I can also use this to replicate recursion
#     # without facing the overhead
#     n = len(d)
#     visited = [False]*n
#     dp = [0]*n
#     finished = [False]*n
#     stack = [start]
#     while stack:
#         start = stack[-1]
#         # stack.pop() # use this if there is nothing after returning
#         if not visited[start]:
#             visited[start] = True
#             for child in d[start]:
#                 if not visited[child]:
#                     stack.append(child)
#         else:
#             stack.pop()
#             dp[start] += 1
#             for child in d[start]:
#                 if finished[child]:
#                     dp[start] += dp[child]
#             finished[start] = True
#             # remove else if you are doing nothing here
#             # add the stuff that you do post traversel here
#             # and add the finished array
#     return dp

def dijkstra(n,d,start):
    dist = [1<<62]*n
    parents = [0]*n
    dist[start] = 0
    parents[start] = 1
    parents1 = parents[:]
    queue = [(0, start)]
    while queue:
        path_len, v = heappop(queue)
        if path_len == dist[v]:
            for w, edge_len in d[v]:
                new_dist = edge_len+path_len
                if new_dist<dist[w]:
                    dist[w] = new_dist
                    parents[w] = parents[v]
                    parents1[w] = parents1[v] 
                    heappush(queue, (new_dist, w))
                elif new_dist==dist[w]:
                    parents[w] += parents[v]
                    parents1[w] += parents1[v]
                    parents[w] %= MOD1
                    parents1[w] %= MOD2
    return dist,parents,parents1
 
def solve():
    n,m = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    d1 = defaultdict(list)

    for i in range(m):
        u,v,w = list(map(int,sys.stdin.readline().split()))
        d[u].append((v,w))
        d1[v].append((u,w))

    dist,parents,pars = dijkstra(n+1,d,1)
    dist1,parents1,pars1 = dijkstra(n+1,d1,n)

    # print(pars[1],pars1[1])
    # print(pars[n])
    
    ans = [i for i in range(1,n+1) if (dist[i]+dist1[i]==dist[n] and ((parents[i]*parents1[i])%MOD1==parents[n] and (pars[i]*pars1[i])%MOD2==pars[n]))]
    # for i in range(1,n+1):
    #     if dist[i]+dist1[i]==y and parents[i]*parents1[i]==x:
    #         ans.append(str(i))
    
    print(str(len(ans))+'\n'+' '.join(map(str,ans)))
    # print(len(ans))
    # print(*ans)
    #L1 = LII()

    #st = SI()
solve()