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

class State:
    def __init__(self):
        self.next = {}
        self.link = -1
        self.len = 0
        self.first_pos = -1
        self.occurrence = 0

class SuffixAutomaton:
    def __init__(self, s):
        self.s = s
        self.states = [State()]
        self.size = 1
        self.last = 0
        for ch in s:
            self.add(ch)
        self._prepare_occurrences() # comment out if taking time
        self._count_substrings()
    def add(self, ch):
        p = self.last
        cur = self.size
        self.states.append(State())
        self.size += 1
        self.states[cur].len = self.states[p].len + 1
        self.states[cur].first_pos = self.states[cur].len - 1
        self.states[cur].occurrence = 1
        while p != -1 and ch not in self.states[p].next:
            self.states[p].next[ch] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[ch]
            if self.states[p].len + 1 == self.states[q].len:
                self.states[cur].link = q
            else:
                clone = self.size
                self.states.append(State())
                self.size += 1
                self.states[clone].len = self.states[p].len + 1
                self.states[clone].next = self.states[q].next.copy()
                self.states[clone].link = self.states[q].link
                self.states[clone].first_pos = self.states[q].first_pos
                while p != -1 and self.states[p].next[ch] == q:
                    self.states[p].next[ch] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
    def _prepare_occurrences(self):
        order = sorted(range(self.size), key=lambda x: -self.states[x].len)
        for i in order:
            if self.states[i].link != -1:
                self.states[self.states[i].link].occurrence += self.states[i].occurrence
    def _count_substrings(self):
        self.dp = [0] * self.size
        for i in range(self.size):
            self.dp[i] = 1
        order = sorted(range(self.size), key=lambda x: self.states[x].len)
        for u in reversed(order):
            for v in self.states[u].next.values():
                self.dp[u] += self.dp[v]
    def is_substring(self, s):
        current = 0
        for ch in s:
            if ch not in self.states[current].next:
                return False
            current = self.states[current].next[ch]
        return True
    def count_occurrences(self, s):
        current = 0
        for ch in s:
            if ch not in self.states[current].next:
                return 0
            current = self.states[current].next[ch]
        return self.states[current].occurrence
    def count_distinct_substrings(self):
        return sum(self.states[i].len - self.states[self.states[i].link].len for i in range(1, self.size))
    def kth_lex_substring(self, k):
        # kth distinct substring
        result = []
        current = 0
        while k:
            for ch in sorted(self.states[current].next):
                next_state = self.states[current].next[ch]
                if self.dp[next_state] < k:
                    k -= self.dp[next_state]
                else:
                    result.append(ch)
                    k -= 1
                    current = next_state
                    break
        return ''.join(result)
    def enumerate_all_substrings(self):
        result = []
        def dfs(state, path):
            for ch in sorted(self.states[state].next):
                next_state = self.states[state].next[ch]
                result.append(path + ch)
                dfs(next_state, path + ch)
        dfs(0, "")
        return result
    def longest_common_substring(self, t):
        v = 0;l = 0;best = 0;bestpos = 0
        for i in range(len(t)):
            while v and t[i] not in self.states[v].next:
                v = self.states[v].link
                l = self.states[v].len
            if t[i] in self.states[v].next:
                v = self.states[v].next[t[i]]
                l += 1
            if l > best:
                best = l
                bestpos = i
        return t[bestpos - best + 1:bestpos + 1]
    def all_occurrences(self, s):
        current = 0
        for ch in s:
            if ch not in self.states[current].next:
                return []
            current = self.states[current].next[ch]
        positions = []
        def collect(state):
            if self.states[state].occurrence:
                pos = self.states[state].first_pos - len(s) + 1
                positions.append(pos)
            for v in self.states[state].next.values():
                collect(v)
        collect(current)
        return sorted(set(positions))
    def missing_sub(self):
        visited = set()
        q = deque([(0, "")])
        while q:
            state, path = q.popleft()
            for c in map(chr, range(97, 123)):
                if c not in self.states[state].next:
                    return path + c
                next_state = self.states[state].next[c]
                if (next_state, path + c) not in visited:
                    visited.add((next_state, path + c))
                    q.append((next_state, path + c))
        return None

def solve():
    print(SuffixAutomaton(SI()).kth_lex_substring(II()))
    #L1 = LII()
    #st = SI()
solve()