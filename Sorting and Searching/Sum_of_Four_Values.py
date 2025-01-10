import sys
from collections import Counter
def solve():
    L = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))
    L2 = []
    L1 = sorted(list(zip(L1,[i for i in range(L[0])])))
    for i in range(len(L1)):
        if len(L2)>=4 and L2[-4][0]==L1[i][0]:
            continue
        else:
            L2.append((L1[i][0],L1[i][1]))
    L1 = L2
    for k in range(len(L1)-3):
        for i in range(k+1,len(L1)-2):
            start = i+1
            end = len(L1)-1
            while start<end:
                y = L1[k][0]+L1[i][0]+L1[start][0]+L1[end][0]
                if y>L[1]:
                    end -= 1
                elif y<L[1]:
                    start += 1
                else:
                    print(L1[k][1]+1,L1[i][1]+1,L1[start][1]+1,L1[end][1]+1)
                    exit() 
    print("IMPOSSIBLE")
solve()