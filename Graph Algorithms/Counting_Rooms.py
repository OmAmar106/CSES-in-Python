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
    L2 = []
    for i in range(L[0]):
    #L1 = list(map(int, sys.stdin.readline().split()))
        st = sys.stdin.readline().strip()
        L2.append(list(st))
    count = 0
    visited = [[False for i in range(len(L2[0]))] for j in range(len(L2))]

    def ispossible(a,b):
        # print(a,b)
        return 0<=a<len(L2) and 0<=b<len(L2[0])
    
    def dfs(a,b):
        st = [(a,b)]
        visited[a][b] = True
        while st:
            x,y = st.pop()
            L = [[1,0],[0,1],[-1,0],[0,-1]]
            for dx,dy in L:
                if ispossible(x+dx,y+dy) and L2[x+dx][y+dy]=='.' and not visited[x+dx][y+dy]:
                    visited[x+dx][y+dy] = True
                    st.append((x+dx,y+dy))

    for i in range(len(L2)):
        for j in range(len(L2[0])):
            if L2[i][j]=='.' and not visited[i][j]:
                dfs(i,j)
                count += 1
    print(count)
solve()