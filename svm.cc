#include <svm.h>
#include <random>

namespace svm{
SVM::SVM(size_t i_max_, Kernel& kernel_, double C_, double epsilon_)
: kernel(kernel_), i_max(i_max), C(C_), epsilon(epsilon_){}

void SVM::fit(const nn::Tensor& X, const nn::Tensor& y){
    auto n = X.shape[0];
    auto d = X.shape[1];
    auto alpha = nn::Tensor(1,n);
    alpha.zeros();
    auto b = 0.0;
    for(size_t i=0; i<max_iter; ++i){
        auto alpha_prev = alpha;
        auto diff = smo();
        if(diff < epsilon){
            break;
        }
    }
}

nn::Tensor SVM::predict(const nn::Tensor& h){
    return sign(nn::dot(w.t(), X.t() + b));
}

size_t SVM::smo(){
    std::uniform_int_distribution<> dist(0,m);
    for(size_t i=0; i<m; ++i){
        auto fx_i = alpha_i * y_o * kernel(x_i, x) + b;
        auto E_i = fx_i - y;
        if((y_i * E_i < -epsilon && alpha_i < C) || (y_i * E_i > tol && a_i > 0)){
            do{
                auto j = dist(eng);
            }while(j == i);
            auto fx_j = alpha_j * y_j * kernel(x_j, x) + b;
            auto E_j = fx_j - y;
            auto alpha_i_old = alpha_i;
            auto alpha_j_old = alpha_j;
            if(y_i != y_j){
                L = max(0,alpha_j - alpha_i);
                H = min(C,C+alpha_j - alpha_i);
            }
            else{
                L = max(0, alpha_i + alpha_j - C);
                H = min(C, alpha_i + alpha_j)
            }
            if(L==H)
                continue;
            auto k_ij = kernel(x_i, x_j);
            auto k_ii = kernel(x_i, x_i);
            auto k_jj = kernel(x_j, x_j);
            auto eta = 2 * k_ij - k_ii - k_jj;
            if(eta >= 0)
                continue;
            alpha_j -= y_j * (E_i - E_j) / eta;
            if(alpha_j > H){
                alpha_j = H;
            }
            else if(alpha_j < L){
                alpha_j = L;
            }
            if(abs(alpha_j - alpha_j_old) < 10E-5){
                continue;
            }
            alpha_j += y_i * y_j *(alpha_j_old - alpha_j);
            auto b_1 = b - E_i - y_i * (alpha_i - alpha_i_old) * k_ii - y_j * (alpha_j - alpha_j_old) * k_ij;
            auto b_2 = b - E_j - y_i * (alpha_i - alpha_i_old) * k_ij - y_j * (alpha_j - alpha_j_old) * k_jj;
            if(alpha_i > 0 && alpha_i < C){
                b = b_1;
            }
            else if(alpha_j > 0 && alpha_j < C){
                b = b_2;
            }
            else{
                b = (b_1 + b_2)/2;
            }
            n_alpha_changed += 1;
        }
    }
    if(n_alpha_changed == 0){
        passes++;
    }
    else{
        passes = 0;
    }
}

LinearKernel::LinearKernel() : Kernel(){};
nn::Tensor LinearKernel::operator()(const nn::Tensor& a, const nn::Tensor& b){
    return nn::dot(a,b.t());
}

PolyKernel::PolyKernel(size_t degree_) : Kernel(), degree(degree_){}
nn::Tensor PolyKernel::operator()(const nn::Tensor& a, nn::Tensor& b){
    return nn::pow(nn::dot(a, b.t()), degree);
}
}
