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
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

from types import GeneratorType

def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

# @bootstrap
# put this just on top of recursion function to increase the recursion limit

# rather than return now use yield and when function being called inside itself, use yield before the function name
# example usage:
# @bootstrap
# def rec1(L,k,cur,count):
# 	if count>=100000:
# 		yield float('INF')
# 	if cur+k+1>=len(L)-1:
# 		yield L[cur]+2
# 	if cur in d:
# 		yield d[cur]
# 	ans = float('INF')
# 	mini = float('INF')
# 	for i in range(k+1,0,-1):
# 		if L[cur+i]<mini:
# 			ans = min(ans,1+L[cur]+(yield rec1(L,k,cur+i,count+1)))
# 			mini = L[cur+i]
# 	d[cur] = ans
# 	yield ans
# the limit of recursion on cf is 10**6

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    mat = [[-1]*8 for i in range(8)]
    mat[L[1]-1][L[0]-1] = 1

    L5 = [(1, 2), (2, 1), (1, -2), (-2, 1), (-1, 2), (2, -1), (-1, -2), (-2, -1)]
    # visited = [1]

    def ispossible(x,y):
        return 0<=x<8 and 0<=y<8
    
    def isempty(x,y):
        return 0<=x<8 and 0<=y<8 and mat[x][y]==-1
    def getDegree(x,y):
        count = 0
        for dx,dy in L5:
            if isempty(x+dx,y+dy):
                count += 1
        return count

    def next1(x,y):
        mini = -1
        c = 0
        mini1 = (9)
        nx = 0
        ny = 0
        start = random.randint(0, 1000) % 8
        for count in range(0, 8):
            i = (start + count) % 8
            nx = x + L5[i][0]
            ny = y + L5[i][1]
            c = getDegree(nx, ny)
            if ((isempty(nx, ny)) and c<mini1):
                mini = i
                mini1 = c
        nx = x + L5[mini][0]
        ny = y + L5[mini][1]
        mat[nx][ny] = mat[x][y] + 1
        x = nx
        y = ny
        return (x,y)
    # repeatedly go to the edge having the minimum degree and for some reason this works weird

    x,y = L[1]-1,L[0]-1
    for i in range(63):
        x,y = next1(x,y) 
    # @bootstrap
    # def rec(x,y):
    #     if visited[0]==64:
    #         yield True
    #     L6 = []
    #     for dx,dy in L5:
    #         if ispossible(x+dx,y+dy):
    #             tot = 0
    #             for dx1,dy1 in L5:
    #                 if ispossible(x+dx+dx1,y+dy+dy1):
    #                     tot += 1
    #             L6.append((-tot,x+dx,y+dy))
    #     L6.sort()
    #     for total,x,y in L6:
    #         mat[x][y]=visited[0]+1
    #         visited[0] += 1
    #         if (yield rec(x,y)):
    #             yield True
    #         visited[0] -= 1
    #         mat[x][y]=-1
    #     yield False        

    # rec(L[1]-1,L[0]-1)
    for i in mat:
        print(*i)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
