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
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, Graphflow(hlpp,dnc): graphflow, DynamicConnectivity: odc
#Treap: treap
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
# if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

class Node:
    def __init__(self,suff,pref,full,maxi):
        self.suff = suff
        self.pref = pref
        self.full = full
        self.maxi = maxi
class SegmentTree:
    def f(node1,node2):
        return Node(max(node2.suff,node1.suff+node2.full),max(node1.pref,node1.full+node2.pref),node1.full+node2.full,max(node1.maxi,node2.maxi,node2.pref+node1.suff))
    def __init__(self,L):
        self.n = len(L)
        self.tree = [None]*((1<<(self.n.bit_length()+1)))
        self.build(1,0,len(L)-1,L)
    def build(self,node,start,end,L):
        st = [(node,start,end,False)]
        while st:
            node,start,end,flag = st.pop()
            if start>end:
                continue
            elif start==end:
                self.tree[node] = Node(L[start],L[start],L[start],L[start])
                continue
            if not flag:
                st.append((node,start,end,True))
                mid = (start+end)>>1
                st.append((2*node,start,mid,False))
                st.append((2*node+1,mid+1,end,False))
            else:
                self.tree[node] = SegmentTree.f(self.tree[2*node],self.tree[2*node+1])
    def _update(self,node,l,r,pos,val):
        while l!=r:
            mid = (l+r)>>1
            if pos>=mid+1:
                l = mid+1
                node = 2*node+1
            else:
                node *= 2
                r = mid
        self.tree[node] = Node(val,val,val,val)
        while node>1:
            node >>= 1
            self.tree[node] = SegmentTree.f(self.tree[2*node],self.tree[2*node+1])
    def update(self,pos,val):
        self._update(1,0,self.n-1,pos,val)

def solve():
    n,q = LII()
    L = LII()
    seg = SegmentTree(L)
    for _ in range(q):
        ind,r = LII()
        seg.update(ind-1,r)
        print(max(0,seg.tree[1].maxi))

    #L1 = LII()
    #st = SI()
solve()