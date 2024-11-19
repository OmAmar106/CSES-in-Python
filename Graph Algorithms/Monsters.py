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
    L1 = []
    for i in range(L[0]):
        L1.append(sys.stdin.readline().strip())
    dp = [[float('inf')]*L[1] for i in range(L[0])]


    def ispossible(x,y):
        return 0<=x<len(L1) and 0<=y<len(L1[0]) 
    
    def bfs1(L):
        Ld = [[1,0],[0,1],[-1,0],[0,-1]]

        for x,y in L:
            for dx,dy in Ld:
                if ispossible(x+dx,y+dy) and L1[x+dx][y+dy]!='#' and dp[x+dx][y+dy]>dp[x][y]+1:
                    if L1[x+dx][y+dy]=='M':
                        dp[x+dx][y+dy] = 0
                        L.append((x+dx,y+dy))
                        continue
                    dp[x+dx][y+dy] = 1+dp[x][y]
                    L.append((x+dx,y+dy))

    L3 = []
    for i in range(L[0]):
        for j in range(L[1]):
            if L1[i][j]=='M':
                dp[i][j] = 0
                L3.append((i,j))
    
    bfs1(L3)

    def bfs():
        used = [[float('inf')]*L[1] for i in range(L[0])]
        parent = [['P']*L[1] for i in range(L[0])]
        i = -1
        for i1 in range(L[0]):
            for j1 in range(L[1]):
                if L1[i1][j1]=='A':
                    i = i1
                    j = j1
                    break
            if i!=-1:
                break

        parent[i][j]='F'

        used[i][j] = 0
        q = [(i,j)]
        Ld = [[1,0,'D'],[0,1,'R'],[-1,0,'U'],[0,-1,'L']]

        for x,y in q:
            # for i in used:
            #     print(*i)
            for dx,dy,w in Ld:
                if ispossible(x+dx,y+dy) and dp[x+dx][y+dy]>1+used[x][y] and L1[x+dx][y+dy]!='#' and used[x+dx][y+dy]==float('inf'):
                    used[x+dx][y+dy] = used[x][y]+1
                    parent[x+dx][y+dy] = w
                    q.append((x+dx,y+dy))
        return parent,used

    parent,used = bfs()
    
    def func(x,y):
        print("YES")
        print(used[x][y])
        L = []
        while parent[x][y]!='F':
            L.append(parent[x][y])
            k = parent[x][y]
            if k=='D':
                x -= 1
            elif k=='U':
                x += 1
            elif k=='L':
                y += 1
            else:
                y -= 1
        L = L[::-1]
        print(''.join(L))

    # for i in dp:
    #     print(*i)
    for i in range(L[1]):
        if used[0][i]!=float('inf'):
            func(0,i)
            return
        elif used[L[0]-1][i]!=float('inf'):
            func(L[0]-1,i)
            return
        
    for i in range(L[0]):
        if used[i][0]!=float('inf'):
            func(i,0)
            return
        elif used[i][-1]!=float('inf'):
            func(i,L[1]-1)
            return
    print("NO")
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()

solve()