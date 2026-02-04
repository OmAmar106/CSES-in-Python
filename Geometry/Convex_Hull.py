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

## here onwards, it is for intersection only, even the point class is different
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))
    if (val > 0):
        return 1
    elif (val < 0):
        return 2
    else:
        return 0
def doIntersect(p1,q1,p2,q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if ((o1 != o2) and (o3 != o4)):
        return True
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
    return False
##
## lines
# 2d line: ax + by + c = 0  is  (a, b, c)
#		  ax + by + c = 0	 ((a, b, c),
# 3d line: dx + ez + f = 0  is  (d, e, f),
#		  gy + hz + i = 0	  (g, h, i))

def get_2dline(p1, p2):
    if p1 == p2:
        return (0, 0, 0)
    _p1, _p2 = min(p1, p2), max(p1, p2)
    a, b, c = _p2[1] - _p1[1], _p1[0] - _p2[0], _p1[1] * _p2[0] - _p1[0] * _p2[1]
    g = gcd(gcd(a, b), c)
    return (a // g, b // g, c // g)
dist = lambda p1, p2: sum((a - b) * (a - b) for a, b in zip(p1, p2))**0.5
get_line = lambda p1, p2: map(get_2dline, combinations(p1, 2), combinations(p2, 2))
is_parallel = lambda l1, l2: l1[0] * l2[1] == l2[0] * l1[1]
is_same = lambda l1, l2: is_parallel(l1, l2) and (l1[1] * l2[2] == l2[1] * l1[2])
collinear = lambda p1, p2, p3: is_same(get_2dline(p1, p2), get_2dline(p2, p3))
intersect = (lambda l1, l2: None if is_parallel(l1, l2) else (
    (l2[1] * l1[2] - l1[1] * l2[2]) / (l2[0] * l1[1] - l1[0] * l2[1]),
    (l1[0] * l2[2] - l1[2] * l2[0]) / (l2[0] * l1[1] - l1[0] * l2[1]),
))
rotate = lambda p, theta, origin=(0, 0): (
    origin[0] + (p[0] - origin[0]) * math.cos(theta) - (p[1] - origin[1]) * math.sin(theta),
    origin[1] + (p[0] - origin[0]) * math.sin(theta) + (p[1] - origin[1]) * math.cos(theta),
)
##
## polygons
dist = lambda p1, p2: sum((a - b) * (a - b) for a, b in zip(p1, p2))**0.5
perimeter = lambda *p: sum(dist(i, j) for i, j in zip(p, p[1:] + p[:1]))
area = lambda *p: abs(sum(i[0] * j[1] - j[0] * i[1] for i, j in zip(p, p[1:] + p[:1]))) / 2
is_in_circle = lambda p, c, r: sum(i * i - j * j for i, j in zip(p, c)) < r * r
incircle_radius = lambda a, b, c: area(a, b, c) / (perimeter(a, b, c) / 2)
circumcircle_radius = lambda a, b, c: (dist(a, b) * dist(b, c) * dist(c, a)) / (4 * area(a, b, c))
##

def convex_hull(P):
    if len(P)<=1:return P
    cross = lambda a,b,c: (a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0])
    P = sorted(P)
    def f(P):
        lower = []
        for p in P:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                lower.pop()
            lower.append(p)
        return lower
    return f(P)[:-1]+f(P[::-1])[:-1]

def solve():
    L = []
    for _ in range(II()):
        L.append(tuple(LII()))

    ans = convex_hull(L)
    print(len(ans))
    for i in ans:
        print(*i)

    #L1 = LII()
    #st = SI()
solve()