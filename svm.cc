#include <svm.hpp>
#include <random>
#include <iostream>


namespace svm{
Kernel LinearKernel = [](const arma::rowvec& a, const arma::rowvec& b){ return arma::dot(a.t(), b); };

SVM::SVM(const Kernel& kernel_, size_t i_max, double C_, double epsilon_)
: kernel(kernel_), i_max(i_max), C(C_), epsilon(epsilon_){
}
SVM::SVM(std::string kernel_, size_t i_max, double C_, double epsilon_):
    i_max(i_max), C(C_), epsilon(epsilon_){
        if(kernel_ == "linear"){
            kernel = LinearKernel;
        }
        else{
            throw std::invalid_argument("Invalid kernel");
        }
    }


void SVM::fit(const arma::mat& x, const arma::vec& y){
    b = 0;
    X = x;
    Y = y;
    m = X.n_rows;
    alpha.set_size(m);
    alpha.zeros();
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> r_dist(0,m-1);
    size_t it = 0;
    while(it < i_max){
        size_t n_alphas = 0;
        for(size_t i=0; i<m; ++i){
            auto E_i = f(X.row(i)) - y(i);
            if(ktt_broken(E_i, i)){
                size_t j;
                // alpha selection (a more complex heuristic could be used, but for small datasets random selection with m(m-1) potential combinations is viable)
                do{
                    j = r_dist(mt);
                }while(j==i);
                auto E_j= f(X.row(j)) - y(j);
                auto a_i_old = alpha(i);
                auto a_j_old = alpha(j);
                // Finding the bounds for alpha
                double L, H;
                if(y(i) != y(j)){
                    L = fmax(0, alpha(j) - alpha(i));
                    H = fmin(C, C + alpha(j) - alpha(i));
                }
                else{
                    L = fmax(0, alpha(i) + alpha(j) - C);
                    H = fmin(C, alpha(i) + alpha(j));
                }
                if(L == H){
                    continue;
                }
                auto k_ii = kernel(X.row(i), X.row(i));
                auto k_ij = kernel(X.row(i), X.row(j));
                auto k_jj = kernel(X.row(j), X.row(j));
                // Optimise alphas as to maximise objective funtion
                auto eta = 2 * k_ij - k_ii - k_jj;
                if(eta >= 0){
                    continue;
                }
                alpha(j) -= y(j) * (E_i - E_j) / eta;
                // Clip to bounds
                if(alpha(j) > H){
                    alpha(j) = H;
                }
                else if(alpha(j) < L){
                    alpha(j) = L;
                }
                if(std::abs(alpha(j) - a_j_old) < 10E-5){
                    continue;
                }
                alpha(i) += y(i) * y(j) * (a_j_old - alpha(j));
                //Threshhold computation so KKT is satified for new alphas
                auto b_1 = b - E_i - y(i) * (alpha(i) - a_i_old) * k_ii - y(i) * (alpha(j) - a_j_old) * k_ij;
                auto b_2 = b - E_j - y(i) * (alpha(i) - a_i_old) * k_ij - y(j) * (alpha(j) -a_j_old) * k_jj;
                if(0 < alpha(i) && alpha(i) < C){
                    b = b_1;
                }
                else if (0 < alpha(j) && alpha(j) < C){
                    b = b_2;
                }
                else{
                    b = (b_1 + b_2) /2;
                }
                n_alphas++;
            }
        }
        if(n_alphas == 0){
            it++;
        }
        else{
            it = 0;
        }
    }
}

arma::vec SVM::predict(const arma::mat& x){
    size_t n = x.n_rows;
    arma::vec p(n);
    for(size_t i=0; i<n; ++i){
        p(i) = sign(f(x.row(i)));
    }
    return p;
}

double SVM::score(const arma::mat& x, const arma::vec& y){
    size_t n = x.n_rows;
    size_t s = 0;
    auto P = predict(x);
    for(size_t i=0; i<n; ++i){
        if(P(i) == y(i)){
            s += 1;
        }
    }
    return s / n;
}

bool SVM::ktt_broken(double E, size_t i){
   return (Y(i) * E < - epsilon && alpha(i) < C) || (Y(i) * E > epsilon && alpha(i) > 0);
}

template<typename T>
int SVM::sign(T x){
    return (T(0) < x) - (x < T(0));
}

double SVM::f(const arma::rowvec& x){
    double F = 0;
    for(size_t i=0; i<m; ++i){
        F += alpha(i) * Y(i) * kernel(X.row(i), x);
    }
    F += b;
    return F;
}
}
