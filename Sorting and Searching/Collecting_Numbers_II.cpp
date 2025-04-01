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
// Segment Tree(lazy propogation): SegmentTree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void solve() {
    int n,q; cin >> n>>q;
    vector<int> L(n);
    for (int &x : L) cin >> x;
    int isprob[n+1] = {0};
    int ans = 1;
    int d[n+1];
    for(int i=0;i<len(L);i++){
        d[L[i]] = i;
    }
    for(int i=1;i<n;i++){
        if(d[i]>d[i+1]){
            ans++;
            isprob[i] = true;
        }
    }
    while(q--){
        int u,v;
        cin>>u>>v;
        u--;
        v--;
        if(isprob[L[v]]){
            isprob[L[v]] = false;
            ans--;
        }
        if(isprob[L[u]]){
            isprob[L[u]] = false;
            ans--;
        }
        if(isprob[L[u]-1]){
            isprob[L[u]-1] = false;
            ans--;
        }
        if(isprob[L[v]-1]){
            isprob[L[v]-1] = false;
            ans--;
        }

        int temp = L[u];
        L[u] = L[v];
        L[v] = temp;
        d[L[u]] = u;
        d[L[v]] = v;

        if(L[u]+1<=n && d[L[u]]>d[L[u]+1] && !isprob[L[u]]){
            isprob[L[u]] = true;
            ans++;
        }
        if(L[v]+1<=n && d[L[v]]>d[L[v]+1] && !isprob[L[v]]){
            isprob[L[v]] = true;
            ans++;
        }
        if(L[u]!=1 && d[L[u]-1]>d[L[u]] && !isprob[L[u]-1]){
            isprob[L[u]-1] = true;
            ans++;
        }
        if(L[v]!=1 && d[L[v]-1]>d[L[v]] && !isprob[L[v]-1]){
            isprob[L[v]-1] = true;
            ans++;
        }
        printf(ans);
    }
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}