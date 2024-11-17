import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append(L1[1])
        d[L1[1]].append(L1[0])
    visited = [False for i in range(L[0]+1)]
    prev = -1
    ans = []
    def dfs(i):
        st = [i]
        while st:
            f = st.pop()
            for i in d[f]:
                if not visited[i]:
                    visited[i] = True
                    st.append(i)
    for i in range(1,L[0]+1):
        if not visited[i]:
            visited[i] = True
            dfs(i)
            if prev!=-1:
                ans.append((prev,i))
            prev = i
    print(len(ans))
    for i in ans:
        print(*i)
    #st = sys.stdin.readline().strip()
solve()