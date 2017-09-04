#ifndef SVM_H
#define SVM_H
#include <armadillo>
#include <string>

namespace svm{

typedef std::function<double(const arma::rowvec&, const arma::rowvec&)> Kernel;

class SVM{
public:
    SVM(const Kernel&, size_t=100, double=1, double=10E-5);
    SVM(std::string, size_t=100, double=1, double=10E-5);
    void fit(const arma::mat& X, const arma::vec& y);
    arma::vec predict(const arma::mat& X);
    double score(const arma::mat& X, const arma::vec& y);
private:
    double f(const arma::rowvec&);
    bool ktt_broken(double, size_t);
    template<typename T>int sign(T);
    Kernel kernel;
    const size_t i_max;
    const double C, epsilon;
    size_t m;
    double b;
    arma::vec alpha, Y;
    arma::mat X;
};
} // namespace svm
#endif // SVM_H
