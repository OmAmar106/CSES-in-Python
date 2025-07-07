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

int mod_exp(int base, int exp, int mod = MOD) {
    int res = 1;
    while (exp) {
        if (exp % 2) res = res * base % mod;
        base = base * base % mod;
        exp /= 2;
    }
    return res;
}

vector<vector<int>> matmul(const vector<vector<int>> &A, const vector<vector<int>> &B) {
    int n = A.size(), m = B[0].size(), p = B.size();
    vector<vector<int>> ans(n, vector<int>(m, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                ans[i][j] = (ans[i][j] + A[i][k] * B[k][j]) % MOD;
            }
        }
    }
    return ans;
}

vector<vector<int>> matpow(vector<vector<int>> M, int power) {
    int size = M.size();
    vector<vector<int>> result(size, vector<int>(size, 0));
    for (int i = 0; i < size; ++i) result[i][i] = 1;
    while (power) {
        if (power % 2 == 1) result = matmul(result, M);
        M = matmul(M, M);
        power /= 2;
    }
    return result;
}

vector<int> sieve(int n) {
    vector<int> primes;
    vector<bool> isp(n + 1, true);
    isp[0] = isp[1] = false;
    for (int i = 2; i <= n; ++i) {
        if (isp[i]) {
            primes.push_back(i);
            for (int j = i * i; j <= n; j += i) isp[j] = false;
        }
    }
    return primes;
}

bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

vector<int> all_factors(int n) {
    vector<int> small, large;
    for (int i = 1; i * i <= n; i += (n & 1) ? 2 : 1) {
        if (n % i == 0) {
            small.push_back(i);
            if (i != n / i) large.push_back(n / i);
        }
    }
    reverse(large.begin(), large.end());
    small.insert(small.end(), large.begin(), large.end());
    return small;
}

vector<int> sieve_unique(int N) {
    vector<int> mini(N);
    iota(mini.begin(), mini.end(), 0);
    for (int i = 2; i < N; ++i) {
        if (mini[i] == i) {
            for (int j = 2 * i; j < N; j += i) mini[j] = i;
        }
    }
    return mini;
}

vector<int> prime_factors(int k, const vector<int>& Lmini) {
    vector<int> ans;
    while (k != 1) {
        ans.push_back(Lmini[k]);
        k /= Lmini[k];
    }
    return ans;
}

int mod_inverse(int a, int mod = MOD) {
    return mod_exp(a, mod - 2, mod);
}


int gauss(vector<vector<int>>& A, vector<int>& sol, int mod) {
    int m = A.size();
    int n = A[0].size() -1;
    vector<int> where(n, -1);
    int rank = 0;

    for (int col = 0; col < n; ++col) {
        int sel = -1;
        for (int row = rank; row < m; ++row) {
            if (A[row][col] % mod != 0) {
                sel = row;
                break;
            }
        }
        if (sel == -1) continue;

        swap(A[rank], A[sel]);

        int inv = mod_exp(A[rank][col], mod - 2, mod);
        for (int j = col; j <= n; ++j)
            A[rank][j] = 1LL * A[rank][j] * inv % mod;

        for (int row = 0; row < m; ++row) {
            if (row != rank && A[row][col]) {
                int factor = A[row][col];
                for (int j = col; j <= n; ++j) {
                    A[row][j] = (A[row][j] - 1LL * factor * A[rank][j]) % mod;
                    if (A[row][j] < 0) A[row][j] += mod;
                }
            }
        }

        where[col] = rank++;
    }

    for (int row = rank; row < m; ++row) {
        if (A[row][n] % mod != 0) return 0;
    }

    sol.assign(n, 0);
    for (int i = 0; i < n; ++i)
        if (where[i] != -1)
            sol[i] = A[where[i]][n];
    for (int x : sol) cout << x << " ";
    return 1;
}

void solve() {
    int n,m;
    cin>>n>>m;
    vii L1;
    vi L2;
    while(n--){
        L1.pb(LII(m+1));
        L2.pb(L1.back().back());
        // L1.back().pop_back();
    }
    int k = (gauss(L1,L2,1e9+7));
    if(!k){
        cout<<-1;
    }
}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}