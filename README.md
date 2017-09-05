# svm-smo
## Usage
The svm module works with a simple fit(X, y) method and predicts using predict(X)

### `svm::SVM`
##### `SVM.SVM(Kernel kernel, size_t max_it=100, double C=1, double epsilon=10E-5)`
* `kernel`: A `function<double(arma::rowvec,arma::rowvec)>` object which describes the kernel of the constructed SVM object.
* `max_it`: The maximum number of sucessive iterations of unchanged alphas before until the fit is completed.
* `C`: The regularization coefficient of the system.
* `epsilon`: The error threshold for violation of the KKT conditions.

##### `SVM.SVM(string kernel, size_t max_it=100, double C=1, double epsilon=10E-5)`
* `kernel`: A string containing the name of the standard kernel chosen. Current kernel choices are "linear".

##### `SVM.fit(arma::mat x, arma::vec y)`
Fit the SVM model to a provided dataset using sequential minimal optimization
* `X`: The feature set to be fitted of shape (_n\_features_, _n\_training\_samples_)
* `y`: The binary categorisation of the dataset of shape (_n\_training\_samples_)

##### `SVM.predict(arma::mat x)`
Returns a arma::vec of predictions for the provided dataset.
* `X`: A feature set of shape (_n\_features_, _n\_samples_)

##### `SVM.score(arna::mat X, arma::vec y)`
Returns the mean prediction rate of the fitted model for samples X against targets y
* `X`: A feature set of shape (_n\_features_, _n\_samples_) to be predicted.
* `y`: A binary classification vector of shape (_n\_samples_) to be compared to predictions on `X`
