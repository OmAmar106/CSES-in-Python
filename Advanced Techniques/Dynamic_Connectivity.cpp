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

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming


struct RollbackDSU {
    vector<int> parent, sz;
    vector<tuple<int,int,int>> st;
    int components;

    RollbackDSU(int n) {
        parent.resize(n);
        sz.assign(n, 1);
        iota(parent.begin(), parent.end(), 0);
        components = n;
    }

    int find(int x) {
        while (x != parent[x])
            x = parent[x];
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);

        if (a == b) {
            st.emplace_back(-1, -1, -1);
            return;
        }

        components--;

        if (sz[a] > sz[b])
            swap(a, b);

        st.emplace_back(a, b, sz[b]);
        parent[a] = b;
        sz[b] += sz[a];
    }

    void rollback() {
        auto [a, b, s] = st.back();
        st.pop_back();

        if (a == -1) return;

        components++;
        parent[a] = a;
        sz[b] = s;
    }
};

struct ParPersistentDSU {
    vector<int> parent, sz;
    vector<int> time;

    ParPersistentDSU(int n) {
        parent.resize(n);
        sz.assign(n, 1);
        time.assign(n, INT_MAX);
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int node, int version) {
        while (!(parent[node] == node || time[node] > version)) {
            node = parent[node];
        }
        return node;
    }

    bool unite(int a, int b, int t) {
        a = find(a, t);
        b = find(b, t);

        if (a == b)
            return false;

        if (sz[a] > sz[b])
            swap(a, b);

        parent[a] = b;
        time[a] = t;
        sz[b] += sz[a];

        return true;
    }

    bool isConnected(int a, int b, int version) {
        return find(a, version) == find(b, version);
    }
};

void solve() {
    int n,m,k;
    cin>>n>>m>>k;
    int u,v;

    map<pair<int,int>,int> d;

    while(m--){
        cin>>u>>v;
        u--;v--;
        if(u<v){
            swap(u,v);
        }
        d[{u,v}] = 0;
    }

    vector<tuple<int,int,int,int>> ranges;

    int typ;
    range(i,k){
        cin>>typ>>u>>v;
        u--;
        v--;
        if(u<v){
            swap(u,v);
        }
        if(typ==1){
            d[{u,v}] = i+1;
        }
        else{
            ranges.emplace_back(d[{u,v}],i+1,u,v);
            d.erase(d.find({u,v}));
        }
    }

    for(auto [x,y]:d){
        ranges.emplace_back(y,k+1,x.first,x.second);
    }

    static int base = k+1;
    vector<vector<pair<int,int>>> seg(2*base+1);

    for(auto [l,r,u,v]:ranges){
        int left = l+base;
        int right = r+base;
        while(left<right){
            if(left&1){
                seg[left].emplace_back(u,v);
                left++;
            }
            if(right&1){
                right--;
                seg[right].emplace_back(u,v);
            }
            left >>= 1;
            right >>= 1;
        }
    }

    auto rec = [](int cur,auto &rec,vi &ans,vector<vector<pi>> &tree,RollbackDSU &rds){
        if(cur>=len(tree)){
            return;
        }
        for(auto [u,v]:tree[cur]){
            rds.unite(u,v);
            // print(cur,u,v);
        }
        rec(2*cur,rec,ans,tree,rds);
        rec(2*cur+1,rec,ans,tree,rds);
        if(cur-base>=0 && cur-base<ans.size()){
            ans[cur-base] = rds.components;
        }
        for(auto [u,v]:tree[cur]){
            rds.rollback();
        }
    };

    RollbackDSU rds(n);

    vi ans(k+1);
    rec(1,rec,ans,seg,rds);

    for(auto it:ans){
        cout<<it<<" ";
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}