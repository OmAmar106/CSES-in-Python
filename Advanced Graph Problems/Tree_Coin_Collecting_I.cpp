// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vector<int>>;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<typename T>
class is_printable {
    template<typename U> static auto test(int) -> decltype(cout << declval<U>(), true_type{});
    template<typename> static auto test(...) -> false_type;
public: static constexpr bool value = decltype(test<T>(0))::value;
};
template<typename T> enable_if_t<is_printable<T>::value> __print(const T& val) { cout << val; }
template<typename T1, typename T2> void __print(const pair<T1, T2>& p) {
    cout << "("; __print(p.first); cout << ":"; __print(p.second); cout << ")";
}
template<typename T> enable_if_t<!is_printable<T>::value> __print(const T& container) {
    cout << "{"; bool first = true;
    for (const auto& x : container) { if (!first) cout << ", "; __print(x); first = false; }
    cout << "}";
}
void print() { cout << "\n"; }
template<typename T, typename... Args>
void print(const T& t, const Args&... rest) {
    __print(t); if constexpr (sizeof...(rest)) cout << " "; print(rest...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, SortedList: sortl
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

int lca(int a, int b, vector<int>& depth,vii& parent,vector<int>& data,vii& mini){
    if(depth[a]<depth[b]){
        int temp = a;
        a = b;
        b = temp;
    }
    int d = depth[a]-depth[b];
    for(int i=0;i<20;i++){
        if((d>>i)&1){
            a = parent[i][a];
        }
    }
    if(a==b){
        return a;
    }
    for(int i=19;i>=0;i--){
        if(parent[i][a]!=parent[i][b]){
            a = parent[i][a];
            b = parent[i][b];
        }
    }
    return parent[0][a];
}

int distf(int a, int b, vector<int>& depth,vii& parent,vector<int>& data,vii& mini){
    int c = lca(a,b,depth,parent,data,mini);
    int dist = depth[a]+depth[b]-2*depth[c];
    int f = data[c];
    int d = depth[a]-depth[c];
    int x = a;
    for(int i=0;i<20;i++){
        if((d>>i)&1){
            f = min(f,mini[i][x]);
            x = parent[i][x];
        }
    }
    d = depth[b]-depth[c];
    x = b;
    for(int i=0;i<20;i++){
        if((d>>i)&1){
            f = min(f,mini[i][x]);
            x = parent[i][x];
        }
    }
    return dist+2*f;
}

void solve() {
    int n,q;
    cin>>n>>q;

    vi L = LII(n);

    vector<vector<int>> d(n);

    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        u--;v--;
        d[u].pb(v);
        d[v].pb(u);
    }

    vector<int> data(len(L),-1);
    queue<pair<int,int>> bfs;
    for(int i=0;i<len(L);i++){
        if(L[i]){
            data[i] = 0;
            bfs.emplace(i,0);
        }
    }

    while(!bfs.empty()){
        auto &[elem,dist] = bfs.front();
        bfs.pop();
        for(auto j:d[elem]){
            if(data[j]==-1){
                data[j] = dist+1;
                bfs.emplace(j,dist+1);
            }
        }
    }

    n = len(d);
    vector<bool> visited(n,false);
    vi depth(n,0);

    stack<int> st;
    st.push(0);

    vector<int> parent1(n,0);
    while(!st.empty()){
        int start = st.top();
        st.pop();
        if(!visited[start]){
            visited[start] = true;
            for(auto child:d[start]){
                if(!visited[child]){
                    depth[child] = depth[start]+1;
                    parent1[child] = start;
                    st.push(child);
                }
            }
        }
    }

    vector<int> mini1;
    for(int i=0;i<n;i++){
        mini1.pb(min(data[i],data[parent1[i]]));
    }

    vector<vector<int>> mini;
    mini.push_back(mini1);

    vector<vector<int>> parent;
    parent.push_back(parent1);

    for(int i=0;i<20;i++){
        vi newparent(parent.back());
        vi newmini(mini.back());

        parent.pb(newparent);
        mini.pb(newmini);

        for(int j=0;j<n;j++){
            parent[len(parent)-1][j] = parent[len(parent)-2][parent[len(parent)-2][j]];
            mini[len(mini)-1][j] = min(mini[len(mini)-2][j],mini[len(mini)-2][parent[len(parent)-2][j]]);
        }
    }

    while(q--){
        int a,b;
        cin>>a>>b;
        a--;b--;
        cout<<distf(a,b,depth,parent,data,mini)<<endl;
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}