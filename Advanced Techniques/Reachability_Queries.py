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
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = open(r'output.txt','w')

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

def GI(n,m,dirs):
    d = [[] for i in range(n)]
    for i in range(m):
        u,v = LII_C(lambda x:int(x)-1)
        d[u].append(v)
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

def euler_path(d):
    start = [1]
    ans = []
    while start:
        cur = start[-1]
        if len(d[cur])==0:
            ans.append(start.pop())
            continue
        k1 = d[cur].pop()
        d[k1].remove(cur) # if undirected
        start.append(k1)
    return ans

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

def bellman_ford(n, edges, start=0):
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
    return dist

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
    return res

class BitSet:
    ADDRESS_BITS_PER_WORD = 12
    BITS_PER_WORD = 1 << ADDRESS_BITS_PER_WORD
    WORD_MASK = -1
    def __init__(self, sz):
        self.sz = sz
        self.words = [0] * (self._wordIndex(sz - 1) + 1)
    def _wordIndex(self, bitIndex):
        if bitIndex >= self.sz:
            raise ValueError("out of bound index", bitIndex)
        return bitIndex >> BitSet.ADDRESS_BITS_PER_WORD
    def flip(self, bitIndex):
        wordIndex = self._wordIndex(bitIndex)
        self.words[wordIndex] ^= 1 << (bitIndex % BitSet.BITS_PER_WORD)
    def flip_range(self, l, r):
        startWordIndex = self._wordIndex(l)
        endWordIndex = self._wordIndex(r)
        firstWordMask = BitSet.WORD_MASK << (l % BitSet.BITS_PER_WORD)
        rem = (r+1) % BitSet.BITS_PER_WORD
        lastWordMask = BitSet.WORD_MASK if rem == 0 else ~(BitSet.WORD_MASK << rem)
        if startWordIndex == endWordIndex:
            self.words[startWordIndex] ^= (firstWordMask & lastWordMask)
        else:
            self.words[startWordIndex] ^= firstWordMask
            for i in range(startWordIndex + 1, endWordIndex):
                self.words[i] ^= BitSet.WORD_MASK
            self.words[endWordIndex] ^= lastWordMask
    def __setitem__(self, bitIndex, value):
        wordIndex = self._wordIndex(bitIndex)
        if value:
            self.words[wordIndex] |= 1 << (bitIndex % BitSet.BITS_PER_WORD)
        else:
            self.words[wordIndex] &= ~(1 << (bitIndex % BitSet.BITS_PER_WORD))
    def __getitem__(self, bitIndex):
        wordIndex = self._wordIndex(bitIndex)
        return self.words[wordIndex] & (1 << (bitIndex % BitSet.BITS_PER_WORD)) != 0
    def nextSetBit(self, fromIndex):
        wordIndex = self._wordIndex(fromIndex)
        word = self.words[wordIndex] & (BitSet.WORD_MASK << (fromIndex % BitSet.BITS_PER_WORD))

        while True:
            if word != 0:
                return wordIndex * BitSet.BITS_PER_WORD + (word & -word).bit_length() - 1
            wordIndex += 1
            if wordIndex > len(self.words) - 1:
                return -1
            word = self.words[wordIndex]
    def nextClearBit(self, fromIndex):
        wordIndex = self._wordIndex(fromIndex)
        word = ~self.words[wordIndex] & (BitSet.WORD_MASK << (fromIndex % BitSet.BITS_PER_WORD))

        while True:
            if word != 0:
                index = wordIndex * BitSet.BITS_PER_WORD + (word & -word).bit_length() - 1
                return index if index < self.sz else - 1
            wordIndex += 1
            if wordIndex > len(self.words) - 1:
                return -1
            word = ~self.words[wordIndex]
    def lastSetBit(self):
        wordIndex = len(self.words) - 1
        word = self.words[wordIndex]

        while wordIndex >= 0:
            if word != 0:
                return wordIndex * BitSet.BITS_PER_WORD + (word.bit_length() - 1 if word > 0 else  BitSet.BITS_PER_WORD - 1)
            wordIndex -= 1
            word = self.words[wordIndex]
        return -1
    def __str__(self):
        res = []
        st = 0
        while True:
            i = self.nextSetBit(st)
            if i != -1:
                res += [0] * (i - st)
                j = self.nextClearBit(i)
                if j != -1:
                    res += [1] * (j-i)
                    st = j
                else:
                    res += [1] * (self.sz - i)
                    break
            else:
                res += [0] * (self.sz - st)
                break

        return "".join(str(v) for v in res)
    def __repr__(self):
        return "Bitset(%s)" % str(self)
    def __iter__(self):
        for i in self[:]:
            yield i
    def __len__(self):
        return self.sz
    def __or__(self, other):
        if self.sz != other.sz:
            raise ValueError("BitSets must be of equal size")
        res = BitSet(self.sz)
        res.words = [a | b for a, b in zip(self.words, other.words)]
        return res
    def __add__(self, other):
        if self.sz != other.sz:
            raise ValueError("BitSets must be of equal size")
        res = BitSet(self.sz)
        carry = 0
        for i in range(len(self.words)):
            total = self.words[i] + other.words[i] + carry
            res.words[i] = total & BitSet.WORD_MASK
            carry = total >> BitSet.BITS_PER_WORD
        return res
    def __and__(self, other):
        if self.sz != other.sz:
            raise ValueError("BitSets must be of equal size")
        res = BitSet(self.sz)
        res.words = [a & b for a, b in zip(self.words, other.words)]
        return res
    def __xor__(self, other):
        if self.sz != other.sz:
            raise ValueError("BitSets must be of equal size")
        res = BitSet(self.sz)
        res.words = [a ^ b for a, b in zip(self.words, other.words)]
        return res
    def __invert__(self):
        res = BitSet(self.sz)
        res.words = [~a & BitSet.WORD_MASK for a in self.words]
        return res
    def add(self, val):
        self.flip_range(val, self.nextClearBit(val))
    def rem(self, val):
        self.flip_range(val, self.nextSetBit(val))
    def bitcount(self):
        return sum(j.bit_count() for j in self.words)

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
        self._imply(self.negate(x),x)
    def solve(self):
        SCC = find_SCC(self.graph)
        order = [0] * (2 * self.n)
        for i, comp in enumerate(SCC):
            for x in comp:
                order[x] = i
        for i in range(self.n):
            if order[i] == order[self.negate(i)]:
                return False, None
        return True, [+(order[i] > order[self.negate(i)]) for i in range(self.n)]

def solve():
    n,m,q = LII()
    d = GI(n,m,dirs=True)
    L = find_SCC(d)
    
    ans = [None]*len(L)
    f = [-1]*n

    for i in range(len(L)-1,-1,-1):
        dp2 = BitSet(len(L))
        for j in L[i]:
            f[j] = i
        dp2[i] = 1

        for j in L[i]:
            for k in d[j]:
                if ans[f[k]]:
                    dp2 |= ans[f[k]]
                    
        ans[i] = dp2

    for _ in range(q):
        u,v = LII_1()
        print("YES" if ans[f[u]][f[v]] else "NO")

    #L1 = LII()
    #st = SI()
solve()