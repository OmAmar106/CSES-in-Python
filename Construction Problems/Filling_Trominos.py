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


# for h in range(1,10):
#     for j in range(1,min(7,h+1)):
#         dp = [[False]*(1<<j) for _ in range(h+1)]
#         dp[0][0] = True
#         for i in range(1,len(dp)):
#             for x in range(1<<j):
#                 for y in range(1<<j):
#                     if ispos(x,y,j):
#                         dp[i][y] |= dp[i-1][x]
#         if dp[-1][0]:
#             print(i,j)

prev = [[(-1,-1)]*101 for i in range(101)]
prev[0][0] = (0,0)
prev[2][3] = prev[3][2] = (0,0)
prev[5][9] = (0,0)
prev[5][6] = (0,0)
prev[9][5] = (0,0)
prev[6][5] = (0,0)

for i in range(101):
    for j in range(101):
        if i%3==0 and j>=2:
            if prev[i][j-2]!=(-1,-1):
                prev[i][j] = (i,j-2)
        if i%2==0 and j>=3:
            if prev[i][j-3]!=(-1,-1):
                prev[i][j] = (i,j-3)
        if j%3==0 and i>=2:
            if prev[i-2][j]!=(-1,-1):
                prev[i][j] = (i-2,j)
        if j%2==0 and i>=3:
            if prev[i-3][j]!=(-1,-1):
                prev[i][j] = (i-3,j)
        if i%5==0 and j>=6:
            if prev[i][j-6]!=(-1,-1):
                prev[i][j] = (i,j-6)
        if j%5==0 and i>=6:
            if prev[i-6][j]!=(-1,-1):
                prev[i][j] = (i-6,j)

# print(prev[33][87])
# print(prev[33][92])

type2 = [
    [0,0],
    [0,1],
    [1,1]
]

type1 = [
    [0,0,1],
    [0,1,1]
]

type4 = [
    'FFPPU',
    'FLPUU',
    'HLLWW',
    'HHSSW',
    'JJOSY',
    'JOOYY',
]


for i in range(len(type4)):
    type4[i] = list(type4[i])
    # print(type3[i])
    for j in range(len(type4[i])):
        type4[i][j] = 3000+ord(type4[i][j])

type3 = [
    'A B B C C E E G G'.split(),
    'A A B D C F E H G'.split(),
    'L L D D F F I H H'.split(),
    'M L N O O K I I J'.split(),
    'M M N N O K K J J'.split()
]

for i in range(len(type3)):
    for j in range(len(type3[i])):
        type3[i][j] = 2000+ord(type3[i][j])

def solve():
    x,y = LII()
    if prev[x][y]==(-1,-1):
        print("NO")
        return

    count = 0
    grid = [['']*y for i in range(x)]
    while (prev[x][y]!=(0,0)) and x!=0 and y!=0 and not(x==5 and y==9) and not (y==5 and x==9):
        prevx = x
        prevy = y
        x,y = prev[x][y]
        # print(x1,x,y1,y)
        if prevx==x:
            for l in range(prevx):
                if prevy==y+3:
                    for l1 in range(y,y+3):
                        grid[l][l1] = count+type1[l%2][l1-(y)]
                    if l%2==1:
                        count += 20
                elif prevy==y+6:
                    for l1 in range(y,y+6):
                        grid[l][l1] = count+type4[l1-y][l%5]
                    if l%5==4:
                        count += 20
                else:
                    for l1 in range(y,y+2):
                        grid[l][l1] = count+type2[l%3][l1-y]
                    if l%3==2:
                        count += 20
        else:
            for l in range(prevy):
                if prevx==x+3:
                    for l1 in range(x,x+3):
                        grid[l1][l] = count+type2[l1-x][l%2]
                    if l%2==1:
                        count += 20
                elif prevx==x+6:
                    for l1 in range(x,x+6):
                        grid[l1][l] = count+type4[l1-x][l%5]
                    if l%5==4:
                        count += 20
                else:
                    for l1 in range(x,x+2):
                        # print(grid[l][l1],x,y)
                        grid[l1][l] = count+type1[l1-x][l%3]
                    if l%3==2:
                        count += 20

    if (x==5 and y==9) or (y==5 and x==9):
        # print(type3)
        for i in range(5):
            for j in range(9):
                if x==5:
                    grid[i][j] = type3[i][j]
                else:
                    grid[j][i] = type3[i][j]

    colors = {}
    dirs = ((0,1),(1,0),(0,-1),(-1,0))
    dirs8 = ((0,1),(1,0),(0,-1),(-1,0),(-1,1),(1,-1),(1,1),(-1,-1))
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            cur = grid[i][j]
            if cur not in colors:
                used = set()

                for dx, dy in dirs8:
                    ni,nj = i+dx,j+dy

                    if 0<=ni<len(grid) and 0<=nj<len(grid[0]):
                        if grid[ni][nj]==grid[i][j]:

                            for dx1,dy1 in dirs:
                                ni1,nj1 = ni+dx1,nj+dy1
                                if 0<=ni1<len(grid) and 0<=nj1<len(grid[0]):
                                    other = grid[ni1][nj1]
                                    if other!=cur and other in colors:
                                        used.add(colors[other])
                            continue
                        
                        other = grid[ni][nj]
                        if other!=cur and other in colors:
                            used.add(colors[other])
                # print(i,j,used)
                c = 0
                while chr(ord('A')+c) in used:
                    c += 1

                colors[cur] = chr(ord('A') + c)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j] = colors[grid[i][j]]

    print("YES")
    for i in grid:
        print(''.join(i))
        # print(*i)
    #L1 = LII()
    #st = SI()
for _ in range(II()):
    #if os.environ.get('LOCAL'):print('________________'+'Test Case : '+str(_+1)+'________________\n')
    solve()