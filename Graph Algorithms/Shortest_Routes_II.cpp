// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define print(arr) for (auto it : arr){cout<<it<<" ";}cout<<endl;
#define len(arr) arr.size()
#define printf(x) cout << x << endl;
#define printm(map) cout<<"{";for(auto it: map){cout<<it.first<<":"<<it.second<<",";};cout<<"}"<<endl;

// const int MOD = 998244353;
const int MOD = 1e9 + 7;

int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    return a / gcd(a, b) * b;
}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void solve() {
    int n,m,q;
    cin>>n>>m>>q;

    int dist[n+1][n+1];
    
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            dist[i+1][j+1] = INF;
        }
        dist[i+1][i+1] = 0;
    }

    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        dist[a][b] = min(dist[a][b],c);
        dist[b][a] = min(dist[b][a],c);
    }
    // printf(dist[1][4])

    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            for(int k=1;k<=n;k++){
                if(dist[i][k]!=INF && dist[j][i]!=INF){
                    dist[j][k] = min(dist[j][k],dist[j][i]+dist[i][k]);
                }
            }
        }
    }
    // printf(dist[1][4])

    while(q--){
        int a,b;
        cin>>a>>b;
        printf((((dist[a][b])!=INF)?dist[a][b]:-1))
    }
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}