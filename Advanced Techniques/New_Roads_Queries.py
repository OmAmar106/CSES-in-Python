import sys,os
from io import BytesIO, IOBase

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
##

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lp,selfop): SegmentTree, bootstrap: bootstrap, Trie: tries
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, Segment Tree(lp): SegmentOther
#Graph1(dnc,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphother
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class PersistentDSU:
    def __init__(self,n):
        self.parent = list(range(n))
        self.size = [1]*n
        self.time = [float('inf')]*n
    def find(self,node,version):
        # returns root at given version
        while not (self.parent[node]==node or self.time[node]>version):
            node = self.parent[node]
        return node
    def union(self,a,b,time):
        # merges a and b 
        a = self.find(a,time)
        b = self.find(b,time)
        if a==b:
            return False
        if self.size[a]>self.size[b]:
            a,b = b,a
        self.parent[a] = b
        self.time[a] = time
        self.size[b] += self.size[a]
        return True
    def isconnected(self,a,b,time):
        return self.find(a,time)==self.find(b,time)

def solve():
    n,m,q = list(map(int, sys.stdin.readline().split()))
    pds = PersistentDSU(n)
    
    for i in range(m):
        u,v = list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
        pds.union(u,v,i+1)
    
    for i in range(q):
        
        u,v = list(map(lambda x:int(x)-1, sys.stdin.readline().split()))

        start = 0
        end = m

        ans = -1

        if m>=3 and not pds.isconnected(u,v,m-3):
            start = m-2

        while start<=end:
            mid = (start+end)//2
            if pds.isconnected(u,v,mid):
                ans = mid
                end = mid-1
            else:
                start = mid+1
        
        print(ans)

    #L1 = LII()
    #st = SI()
solve()