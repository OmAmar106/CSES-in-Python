// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define int long long
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vi>;
using viii = vector<vii>;
using ll = long long;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#ifdef LOCAL
#define dbg(x...) cout<<"["<<#x<<"] = "; print(x);
#else
#define dbg(x...)
#endif

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<class T> bool chmin(T& a,const T& b){ if(b<a){ a=b; return 1;} return 0; }
template<class T> bool chmax(T& a,const T& b){ if(a<b){ a=b; return 1;} return 0;}
template<class T> using minpq = priority_queue<T,vector<T>,greater<T>>;
template<class T> using maxpq = priority_queue<T>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<class T> void _print(const T &x){ cout << x; }
template<class T,class U> void _print(const pair<T,U> &p){cout<<"(";_print(p.first);cout<<",";_print(p.second);cout<<")";}
template<class T>
void _print(const vector<T> &v){
    cout<<"[";bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"]";
}
template<class T>
void _print(const set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T>
void _print(const unordered_set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const unordered_map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
void print(){ cout<<"\n"; }
template<class T,class... Args>
void print(const T& a,const Args&... args){
    _print(a);
    if constexpr(sizeof...(args)) cout<<" ";
    print(args...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}
int rnd(int l,int r){return uniform_int_distribution<int>(l,r)(rng);}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Rollback/Par DSU: rbdsu, treap: treap, graphflow(mat_match): graphflow
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

struct PersistentSegmentTree {
    // Everything is 0 based indexing and [l,r]
    struct Node {
        int value;
        Node* left;
        Node* right;

        Node(int v = 0, Node* l = nullptr, Node* r = nullptr) {
            value = v;
            left = l;
            right = r;
        }
    };

    int n;
    vector<Node*> versions;

    static int func(int a, int b) {
        return a + b;
    }

    Node* build(int l,int r,vector<int>& data){
        if(l==r) return new Node(data[l]);
        int mid=(l+r)/2;
        Node* left=build(l,mid,data);
        Node* right=build(mid+1,r,data);
        return new Node(func(left->value,right->value),left,right);
    }

    PersistentSegmentTree(vector<int>& data) {
        n = data.size();
        versions.push_back(build(0,n-1,data));
    }

    Node* update(int version, int pos, int value) {
        // Returns the root to the new tree,
        // now you must add it to versions yourself
        // seg.versions[version] = seg.update(version, pos, value)

        function<Node*(Node*,int,int)> dfs = [&](Node* node,int left,int right)->Node*{
            if(left==right){
                int k=value;
                // int k = func(node->value, value); // if i want to update
                return new Node(k);
            }
            int mid=(left+right)/2;
            if(pos<=mid){
                Node* l=dfs(node->left,left,mid);
                return new Node(func(l->value,node->right->value),l,node->right);
            }else{
                Node* r=dfs(node->right,mid+1,right);
                return new Node(func(node->left->value,r->value),node->left,r);
            }
        };

        return dfs(versions[version],0,n-1);
    }

    int create_version(int version, int pos, int value) {
        Node* new_root = update(version, pos, value);
        versions.push_back(new_root);
        return (int)versions.size() - 1;
    }

    int query(int version, int ql, int qr) {
        Node* node = versions[version];

        function<int(Node*,int,int)> dfs = [&](Node* node,int left,int right)->int{
            if(ql>right||qr<left) return 0;
            if(ql<=left&&right<=qr) return node->value;
            int mid=(left+right)/2;
            return func(
                dfs(node->left,left,mid),
                dfs(node->right,mid+1,right)
            );
        };

        return dfs(node,0,n-1);
    }
};

void solve() {
    int n,q;
    cin>>n>>q;
    vi L(n);
    for(auto &i:L){
        cin>>i;
    }

    PersistentSegmentTree perseg = PersistentSegmentTree(L);
    int k3 = 0;
    while(q--){
        int typ;
        cin>>typ;
        if(typ==1){
            int k,a,x;
            cin>>k>>a>>x;
            k--;
            perseg.versions[k] = perseg.update(k,a-1,x);
        }
        else if(typ==2){
            int k,a,b;
            cin>>k>>a>>b;
            k--;
            print(perseg.query(k,a-1,b-1));
        }
        else{
            int k;
            cin>>k;
            k--;
            int a = 0;
            perseg.create_version(k,a,(perseg.query(k,a,a)));
        }   
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}