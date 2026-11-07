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
# II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
# MI = lambda x : x(map(int, sys.stdin.readline().split()))
SI = lambda : sys.stdin.readline().strip()
# SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
# LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
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
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode, mcmf: mcmf, pref2d: pref2d, RecSegTree: SegmentTreeRec, AVL sortedlist: asortl
#bwt: bwt, linalg: linalg, LCT: lct
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());
 
# def DAG_to_SCC(graph):
#     indeg = [0]*len(graph)
#     outdeg = [len(graph[i]) for i in range(len(graph))]
#     for i in range(len(graph)):
#         for j in graph[i]:
#             indeg[j] += 1
#     n = len(graph)
#     if n <= 1:
#         return []
#     visited = [False]*n
#     def dfs(node):
#         gr = graph
#         stack = [(node,0)]
#         visited[node] = True
#         while stack:
#             start,ind = stack[-1]
#             if outdeg[start]==0:
#                 return start
#             if ind<len(graph[start]):
#                 stack[-1] = (start,ind+1)
#                 if not visited[gr[start][ind]]:
#                     visited[gr[start][ind]] = True
#                     stack.append((gr[start][ind],0))
#             else:
#                 stack.pop()
#         return None
#     matched_sources = []
#     matched_sinks = []
#     never_matched_sources = []
#     never_matched_sinks = []
#     for node in range(n):
#         if indeg[node] == 0:
#             matching_sink = dfs(node)
#             if matching_sink is not None:
#                 matched_sources.append(node)
#                 matched_sinks.append(matching_sink)
#             else:
#                 never_matched_sources.append(node)
#     for node in range(n):
#         if outdeg[node] == 0 and not visited[node]:
#             never_matched_sinks.append(node)
#     sources = matched_sources + never_matched_sources
#     sinks = matched_sinks + never_matched_sinks
#     P = len(matched_sources)
#     N = len(sources)
#     M = len(sinks)
#     K = min(N, M)
#     ret =  [(0, P - 1)]
#     ret += ((i, i - 1) for i in range(1, P))
#     ret += ((i, i)	 for i in range(P, K))
#     ret += ((0, i)	 for i in range(K, N))
#     ret += ((i, 0)	 for i in range(K, M))
#     return [(sinks[i], sources[j]) for i,j in ret]
 
# def kahn(graph):
#     n = len(graph)
#     indeg, idx = [0] * n, [0] * n
#     for i in range(n):
#         for e in graph[i]:
#             indeg[e] += 1
#     q, res = [], []
#     for i in range(n):
#         if indeg[i] == 0:
#             q.append(i)
#     nr = 0
#     while q:
#         res.append(q.pop())
#         idx[res[-1]], nr = nr, nr + 1
#         for e in graph[res[-1]]:
#             indeg[e] -= 1
#             if indeg[e] == 0:
#                 q.append(e)
#     return res, idx, nr == n
 
def find_SCC(graph):
    SCC, S, P = [], [], []
    depth = [0] * len(graph)
    stack = list(range(len(graph)))
    while stack:
        node = stack.pop()
        if node < 0:
            d = depth[~node] - 1
            if P[-1] > d:
                SCC.append(S[d:])
                del S[d:], P[-1]
                for node in SCC[-1]:
                    depth[node] = -1
        elif depth[node] > 0:
            while P[-1] > depth[node]:
                P.pop()
        elif depth[node] == 0:
            S.append(node)
            P.append(len(S))
            depth[node] = len(S)
            stack.append(~node)
            stack += graph[node]
    return SCC[::-1]
 
class TwoSat:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(2 * n)]
    def negate(self, x):
        return x+self.n if x<self.n else x-self.n
    def _imply(self, x, y):
        # agar x hoga , toh y hoga
        self.graph[x].append(y)
        self.graph[self.negate(y)].append(self.negate(x))
    def either(self, x, y):
        # koi ek true ho sakta hain ya dono bhi
        self._imply(self.negate(x),y)
        self._imply(self.negate(y),x)
    def set(self, x):
        self._imply(self.negate(x)-self.n,x)
    def solve(self):
        SCC = find_SCC(self.graph)
        order = [0] * (2 * self.n)
        for i, comp in enumerate(SCC):
            for x in comp:
                order[x] = i
        for i in range(self.n):
            if order[i] == order[i+self.n]:
                return False, None
        return True, [+(order[i] > order[i+self.n]) for i in range(self.n)]
 
def solve():
    n,m = LII()
 
    L = []
    for i in range(n):
        L.extend(list(SI()))
 
    sat = TwoSat(3*n*m)
 
    def f2(pos1):
        cur1 = ord(L[pos1])-ord('A')
        pos1 *= 3
        # sat.set(sat.negate(pos1+cur1))
 
        def f1(ind,val):
            if val==0:
                sat._imply(ind+1,sat.negate(ind+2))
                sat._imply(sat.negate(ind+2),ind+1)
            elif val==1:
                sat._imply(ind,sat.negate(ind+2))
                sat._imply(sat.negate(ind+2),ind)
            else:
                sat._imply(ind+1,sat.negate(ind))
                sat._imply(sat.negate(ind),ind+1)
                            
 
        f1(pos1,cur1)
 
    def medge(pos1,pos2):
        pos1 *= 3
        pos2 *= 3
 
        for j in range(3):
            sat._imply(pos1+j,sat.negate(pos2+j))
 
    for i in range(n-1):
        for j in range(m-1):
            cur = m*i+j
            medge(cur,cur+m)
            medge(cur,cur+1)
            f2(cur)
 
    for i in range(n-1):
        cur = (i+1)*m-1
        medge(cur,cur+m)
        f2(cur)
 
    for j in range(m-1):
        cur = (n-1)*m+j
        medge(cur,cur+1)
        f2(cur)
 
    f2(n*m-1)
 
    f,ans = sat.solve()
    L = ans
 
    if f:
        ans = []
        for i in range(n):
            for j in range(m):
                if L[(3*(i*(m)+j))]:
                    ans.append('A')
                elif L[(3*(i*(m)+j))+1]:
                    ans.append('B')
                else:
                    ans.append('C')
            ans.append('\n')
        print(''.join(ans))
 
    else:
        print("IMPOSSIBLE")
 
    #L1 = LII()
    #st = SI()
solve()
