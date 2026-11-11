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

class KMPAutomaton:
    def __init__(self, pattern, alphabet=26):
        # Build KMP automaton for given pattern O(n * A)
        self.p = pattern
        self.n = len(pattern)
        self.A = alphabet
        self.pi = self._build_pi()
        self.aut = self._build_automaton()
    def _build_pi(self):
        pi = [0] * self.n
        j = 0
        for i in range(1, self.n):
            while j > 0 and self.p[i] != self.p[j]:
                j = pi[j - 1]
            if self.p[i] == self.p[j]:
                j += 1
            pi[i] = j
        return pi
    def _build_automaton(self):
        aut = [[0] * self.A for _ in range(self.n + 1)]
        s = self.p + '#'
        for i in range(self.n + 1):
            for c in range(self.A):
                ch = chr(ord('a') + c)
                if i > 0 and ch != s[i]:
                    aut[i][c] = aut[self.pi[i - 1]][c]
                else:
                    aut[i][c] = i + (ch == s[i])
        return aut
    def next_state(self, state, ch):
        return self.aut[state][ord(ch) - ord('a')]
    def count_avoiding(self, length, mod=10**9+7):
        # Count strings of given length that DO NOT contain pattern O(length * n * A)
        dp = [0] * self.n
        dp[0] = 1
        for _ in range(length):
            new_dp = [0] * self.n
            for state in range(self.n):
                if dp[state] == 0:
                    continue
                for c in range(self.A):
                    nxt = self.aut[state][c]
                    if nxt == self.n:
                        continue
                    new_dp[nxt] = (new_dp[nxt] + dp[state]) % mod
            dp = new_dp
        return sum(dp) % mod
    def count_containing(self, length, mod=10**9+7):
        # Count strings of given length that contain pattern at least once O(length * n * A)
        total = pow(self.A, length, mod)
        bad = self.count_avoiding(length, mod)
        return (total - bad) % mod
    def max_occurrences(self, length):
        # Max number of times pattern can appear in any string of given length O(length * n * A)
        dp = [-10**9] * self.n
        dp[0] = 0
        for _ in range(length):
            new_dp = [-10**9] * self.n
            for state in range(self.n):
                if dp[state] < 0:
                    continue
                for c in range(self.A):
                    nxt = self.aut[state][c]
                    gain = 1 if nxt == self.n else 0
                    if nxt == self.n:
                        nxt = self.pi[self.n - 1]
                    new_dp[nxt] = max(new_dp[nxt], dp[state] + gain)
            dp = new_dp
        return max(dp)
    def contains(self, text):
        # Check if pattern exists in given text
        state = 0
        for ch in text:
            state = self.next_state(state, ch)
            if state == self.n:
                return True
        return False

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
        # self._count_substrings()
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
        self.sub_occ = [0] * self.size

        for v in order:
            self.sub_occ[v] = self.states[v].occurrence
            for u in self.states[v].next.values():
                self.sub_occ[v] += self.sub_occ[u]

def solve():
    st = SI()
    k = II()

    sa = SuffixAutomaton(st)

    # print(sa.sub_occ[sa.states[0].next['a']])

    cur = 0
    ans = []
    states = sa.states

    for i in range(len(st)):
        # print(ans)
        if i!=0:
            val = sa.sub_occ[cur]
            for j in range(26):
                try:
                    z = sa.sub_occ[states[cur].next[chr(ord('a')+j)]]
                    val -= z
                except:
                    pass
            # print(val,k,ans)
            if val>=k:
                break
            k -= val
        for j in range(26):
            try:
                z = sa.sub_occ[states[cur].next[chr(ord('a')+j)]]
                if z<k:
                    # print(k,z,j)
                    k -= z
                elif z>=k:
                    ans.append(chr(ord('a')+j))
                    cur = sa.states[cur].next[ans[-1]]
                    break
            except:
                pass

    print(''.join(ans))
    
    #L1 = LII()
    #st = SI()
solve()