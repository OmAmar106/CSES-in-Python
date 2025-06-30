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
#2-D BIT: 2DBIT, MonoDeque: mono
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

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
        # print([(a & b) for a, b in zip(self.words, other.words)])
        return sum([bin(a & b).count('1') for a, b in zip(self.words, other.words)])
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

def solve():
    n = II()
    L = []
    ans = 0
    for i in range(n):
        st = SI()[::-1]
        L2 = BitSet(n)
        for j in range(len(st)):
            if st[j]=='1':
                L2[j] = 1
        # print(L2)
        for j in range(i):
            x = L2&L[j]
            ans += x*(x-1)//2
        L.append(L2)

    print(ans)

    #L1 = LII()
    #st = SI()
solve()