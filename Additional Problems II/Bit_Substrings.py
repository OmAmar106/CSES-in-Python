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
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

class FastFFT:
    # This is a bit faster, one log n factor is less but it's accuracy is not 100%
    # use this when coeffecient does not matter(set coeffecient to 1 repeatedly)
    # or you could just risk it :)
    def __init__(self, MOD=998244353):
        FastFFT.MOD = MOD
        # g = self.primitive_root_constexpr()
        g = 3
        ig = pow(g, FastFFT.MOD - 2, FastFFT.MOD)
        FastFFT.W = [pow(g, (FastFFT.MOD - 1) >> i, FastFFT.MOD) for i in range(30)]
        FastFFT.iW = [pow(ig, (FastFFT.MOD - 1) >> i, FastFFT.MOD) for i in range(30)]
    def primitive_root_constexpr(self):
        if FastFFT.MOD == 998244353:
            return 3
        elif FastFFT.MOD == 200003:
            return 2
        elif FastFFT.MOD == 167772161:
            return 3
        elif FastFFT.MOD == 469762049:
            return 3
        elif FastFFT.MOD == 754974721:
            return 11
        divs = [0] * 20
        divs[0] = 2
        cnt = 1
        x = (FastFFT.MOD - 1) // 2
        while x % 2 == 0:
            x //= 2
        i = 3
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1
                while x % i == 0:
                    x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (FastFFT.MOD - 1) // divs[i], FastFFT.MOD) == 1:
                    ok = False
                    break
            if ok:
                return g
            g += 1
    def fft(self, k, f):
        for l in range(k, 0, -1):
            d = 1 << l - 1
            U = [1]
            for i in range(d):
                U.append(U[-1] * FastFFT.W[l] % FastFFT.MOD)
            for i in range(1 << k - l):
                for j in range(d):
                    s = i * 2 * d + j
                    f[s], f[s + d] = (f[s] + f[s + d]) % FastFFT.MOD, U[j] * (f[s] - f[s + d]) % FastFFT.MOD
    def ifft(self, k, f):
        for l in range(1, k + 1):
            d = 1 << l - 1
            for i in range(1 << k - l):
                u = 1
                for j in range(i * 2 * d, (i * 2 + 1) * d):
                    f[j+d] *= u
                    f[j], f[j + d] = (f[j] + f[j + d]) % FastFFT.MOD, (f[j] - f[j + d]) % FastFFT.MOD
                    u = u * FastFFT.iW[l] % FastFFT.MOD
    def convolve(self, A, B):
        n0 = len(A) + len(B) - 1
        k = (n0).bit_length()
        n = 1 << k
        A += [0] * (n - len(A))
        B += [0] * (n - len(B))
        self.fft(k, A)
        self.fft(k, B)
        A = [a * b % FastFFT.MOD for a, b in zip(A, B)]
        self.ifft(k, A)
        inv = pow(n, FastFFT.MOD - 2, FastFFT.MOD)
        del A[n0:]
        for i in range(n0):
            A[i] = (A[i]*inv)%FastFFT.MOD
        return A

class FFT:
    def __init__(self, MOD=998244353,MOD1=469762049):
        FFT.MOD = MOD
        FFT.MOD1 = MOD1
        FFT.MOD2 = pow(MOD,MOD1-2,MOD1)
        FFT.mod_inv = (self.XT_GCD(MOD,MOD1)[1])%MOD1
        # g = self.primitive_root_constexpr()
        g = 3
        ig = pow(g, FFT.MOD - 2, FFT.MOD)
        ig1 = pow(g, FFT.MOD1 - 2, FFT.MOD1)
        FFT.W = [pow(g, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
        FFT.W1 = [pow(g, (FFT.MOD1 - 1) >> i, FFT.MOD1) for i in range(30)]
        FFT.iW = [pow(ig, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
        FFT.iW1 = [pow(ig1, (FFT.MOD1 - 1) >> i, FFT.MOD1) for i in range(30)]
    def primitive_root_constexpr(self):
        if FFT.MOD == 998244353:
            return 3
        elif FFT.MOD == 200003:
            return 2
        elif FFT.MOD == 167772161:
            return 3
        elif FFT.MOD == 469762049:
            return 3
        elif FFT.MOD == 754974721:
            return 11
        divs = [0] * 20
        divs[0] = 2
        cnt = 1
        x = (FFT.MOD - 1) // 2
        while x % 2 == 0:
            x //= 2
        i = 3
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1
                while x % i == 0:
                    x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (FFT.MOD - 1) // divs[i], FFT.MOD) == 1:
                    ok = False
                    break
            if ok:
                return g
            g += 1
    def fft(self, k, f,f1):
        for l in range(k, 0, -1):
            d = 1 << l - 1
            U = [(1,1)]
            for i in range(d):
                U.append((U[-1][0] * FFT.W[l] % FFT.MOD,U[-1][1] * FFT.W1[l] % FFT.MOD1))
            for i in range(1 << k - l):
                for j in range(d):
                    s = i * 2 * d + j
                    f[s], f[s + d] = (f[s] + f[s + d]) % FFT.MOD, U[j][0] * (f[s] - f[s + d]) % FFT.MOD
                    f1[s], f1[s + d] = (f1[s] + f1[s + d]) % FFT.MOD1, U[j][1] * (f1[s] - f1[s + d]) % FFT.MOD1
    def ifft(self, k, f,f1):
        for l in range(1, k + 1):
            d = 1 << l - 1
            for i in range(1 << k - l):
                u = 1
                u1 = 1
                for j in range(i * 2 * d, (i * 2 + 1) * d):
                    f[j+d] *= u
                    f[j], f[j + d] = (f[j] + f[j + d]) % FFT.MOD, (f[j] - f[j + d]) % FFT.MOD
                    u = u * FFT.iW[l] % FFT.MOD
                    f1[j+d] *= u1
                    f1[j], f1[j + d] = (f1[j] + f1[j + d]) % FFT.MOD1, (f1[j] - f1[j + d]) % FFT.MOD1
                    u1 = u1 * FFT.iW1[l] % FFT.MOD1
    def XT_GCD(self,a,b):
        if b == 0:
            return a,1,0
        g,x1,y1 = self.XT_GCD(b,a%b)
        x = y1
        y = x1-(a//b)*y1
        return g,x,y
    def CRT(self,a, mod1, b, mod2):
        k = (a+(b-a)*self.mod_inv%mod2*mod1)%(mod1*mod2)
        return k
    def convolve(self, A, B):
        n0 = len(A) + len(B) - 1
        k = (n0).bit_length()
        n = 1 << k
        A += [0] * (n - len(A))
        B += [0] * (n - len(B))
        A1 = A[:]
        B1 = B[:]
        self.fft(k, A,A1)
        self.fft(k, B,B1)
        A = [a * b % FFT.MOD for a, b in zip(A, B)]
        A1 = [a * b % FFT.MOD1 for a, b in zip(A1, B1)]
        self.ifft(k, A,A1)
        inv = pow(n, FFT.MOD - 2, FFT.MOD)
        inv1 = pow(n, FFT.MOD1 - 2, FFT.MOD1)
        del A[n0:]
        for i in range(n0):
            A[i] = self.CRT(A[i]*inv,FFT.MOD,A1[i]*inv1,FFT.MOD1)
        return A

fft = FFT()

def solve():
    # n = II()
    # L = LII()
    #L1 = LII()
    st = SI()

    pref = [0]
    for i in st:
        pref.append(pref[-1])
        pref[-1] += (i=='1')
    
    arr = [0]*(len(st)+1)
    for i in pref:
        arr[i] += 1
    
    # print(arr)

    # ans[j] = number of subarrays such that pref[r]-pref[l] = j = number of occurence of pref[r]* number of occurence of pref[l]
    # such that pref[r]-pref[l] = j

    ans = fft.convolve(arr[:],arr[::-1])[len(arr)-1:]
    ans[0] = 0 

    d = Counter()
    for i in pref:
        ans[0] += d[i]
        d[i] += 1
        
    print(*ans)


solve()