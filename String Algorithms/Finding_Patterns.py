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
#2-D BIT: 2DBIT, MonoDeque: mono, nummat: matrix
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

def SuffArr(L):
    n = len(L)
    buckets = [0] * (max(L) + 2)
    for a in L:
        buckets[a + 1] += 1
    for b in range(1, len(buckets)):
        buckets[b] += buckets[b - 1]
    isL = [1] * n
    for i in reversed(range(n - 1)):
        isL[i] = +(L[i] > L[i + 1]) if L[i] != L[i + 1] else isL[i + 1]
    def induced_sort(LMS):
        SA = [-1] * (n)
        SA.append(n)
        endpoint = buckets[1:]
        for j in reversed(LMS):
            endpoint[L[j]] -= 1
            SA[endpoint[L[j]]] = j
        startpoint = buckets[:-1]
        for i in range(-1, n):
            j = SA[i] - 1
            if j >= 0 and isL[j]:
                SA[startpoint[L[j]]] = j
                startpoint[L[j]] += 1
        SA.pop()
        endpoint = buckets[1:]
        for i in reversed(range(n)):
            j = SA[i] - 1
            if j >= 0 and not isL[j]:
                endpoint[L[j]] -= 1
                SA[endpoint[L[j]]] = j
        return SA
    isLMS = [+(i and isL[i - 1] and not isL[i]) for i in range(n)]
    isLMS.append(1)
    LMS = [i for i in range(n) if isLMS[i]]
    if len(LMS) > 1:
        SA = induced_sort(LMS)
        LMS2 = [i for i in SA if isLMS[i]]
        prev = -1
        j = 0
        for i in LMS2:
            i1 = prev
            i2 = i
            while prev >= 0 and L[i1] == L[i2]:
                i1 += 1
                i2 += 1
                if isLMS[i1] or isLMS[i2]:
                    j -= isLMS[i1] and isLMS[i2]
                    break
            j += 1
            prev = i
            SA[i] = j
        LMS = [LMS[i] for i in SuffArr([SA[i] for i in LMS])]
    return induced_sort(LMS)

def LCPArray(L,SA=None):
    # Longest Common prefix in between S[i:] and S[i+1:]
    if not SA:
        SA = SuffArr(L)
    n = len(L)
    rank = [0] * n
    for i in range(n):
        rank[SA[i]] = i
    LCP = [0] * (n - 1)
    k = 0
    for i in range(n):
        k = rank[i]
        if k==n-1:
            continue
        j = SA[k+1]
        while i+k<n and L[i+k] == L[j+k]:
            k += 1
        LCP[k] = k
        k -= k > 0
    return LCP

def z_function(S):
    # return: the Z array, where Z[i] = length of the longest common prefix of S[i:] and S
    n = len(S)
    Z = [0] * n
    l = r = 0
    for i in range(1, n):
        z = Z[i - l]
        if i + z >= r:
            z = max(r - i, 0)
            while i + z < n and S[z] == S[i + z]:
                z += 1
            l, r = i, i + z
        Z[i] = z
    Z[0] = n
    return Z

def manacher(s):
    # returns longest palindrome in s
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    L = [0] * n
    c = r = 0
    ml = 0
    mc = 0
    for i in range(n):
        mirror = 2*c-i
        if i<r:
            L[i] = min(r-i,L[mirror])
        a = i+L[i]+1
        b = i-L[i]-1
        while a<n and b>=0 and t[a]==t[b]:
            L[i] += 1
            a += 1
            b -= 1
        if i+L[i]>r:
            c = i
            r = i+L[i]
        if L[i]>ml:
            ml = L[i]
            mc = i
    start = (mc-ml)//2
    return s[start:start+ml]

def SuffArray(s):
    # Starting position of ith suffix
    # in lexographic order
    s += "$"
    n = len(s)
    k = 0
    rank = [ord(c) for c in s]
    tmp = [0] * n
    sa = list(range(n))
    def sort_key(i):
        return (rank[i], rank[i + (1 << k)] if i + (1 << k) < n else -1)
    while True:
        sa.sort(key=sort_key)
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i-1]] + (sort_key(sa[i-1]) != sort_key(sa[i]))
        rank, tmp = tmp, rank
        k += 1
        if (1 << k) >= n:
            break
    return sa

HMOD = 2147483647
HBASE1 = random.randrange(HMOD)
HBASE2 = random.randrange(HMOD)
class Hashing:
    def __init__(self, s, mod=HMOD, base1=HBASE1, base2=HBASE2):
        self.mod, self.base1, self.base2 = mod, base1, base2
        self._len = _len = len(s)
        f_hash, f_pow = [0] * (_len + 1), [1] * (_len + 1)
        # s_hash, s_pow = f_hash[:], f_pow[:]
        for i in range(_len):
            f_hash[i + 1] = (base1 * f_hash[i] + (s[i])) % mod
            # s_hash[i + 1] = (base2 * s_hash[i] + s[i]) % mod
            f_pow[i + 1] = base1 * f_pow[i] % mod
            # s_pow[i + 1] = base2 * s_pow[i] % mod
        self.f_hash, self.f_pow = f_hash, f_pow
        # self.s_hash, self.s_pow = s_hash, s_pow
    def hashed(self, start, stop):
        return (
            (self.f_hash[stop] - self.f_pow[stop - start] * self.f_hash[start]) % self.mod
            # (self.s_hash[stop] - self.s_pow[stop - start] * self.s_hash[start]) % self.mod,
        )
    def get_hashes(self, length):
        return (
            [(self.f_hash[i + length] - self.f_pow[length] * self.f_hash[i]) % self.mod for i in range(self._len - length + 1)],
            [(self.s_hash[i + length] - self.s_pow[length] * self.s_hash[i]) % self.mod for i in range(self._len - length + 1)],
        )
    
def solve():
    st = SLI()
    st1 = SuffArr(st)

    for i in range(II()):
        q = SLI()
        
        def func(pos):
            for j in range(pos,len(st)):
                if j-pos==len(q):
                    return 2
                if (st[j]!=q[j-pos]):
                    return 1*(st[j]<q[j-pos])
                
            return 2 if len(q)==len(st)-pos else 1
        
        start = 0
        end = len(st1)-1
        flag = False

        while start<=end:
            mid = (start+end)//2
            t = func(st1[mid])
            # print(q,st[st1[mid]:],t)
            if t==2:
                flag = True
                break
            elif t==0:
                end = mid-1
            else:
                start = mid+1
        
        print("YES" if flag else "NO")

    #L1 = LII()
    #st = SI()
solve()