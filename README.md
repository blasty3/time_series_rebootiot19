# Multivariate Time Series Anomaly Detection

Code related to my Master's thesis.

## Structure

| Module              | Description   |
| ------------------- | ------------- |
| `anomaly_detection` | Anomaly detection models and utilities. |
| `data`              | Data files.    |
| `experiments`       | Experiments and samples.    |

## Models

| Model        | Description   |
| ------------ | ------------- |
| DSVAE-AD     | [Disentangled Sequential Variational Autoencoder Based Anomaly Detector](http://urn.fi/URN:NBN:fi:aalto-202101311730) |
| MEWMA        | [Multivariate Exponentially Weighted Moving Average control chart](https://amstat.tandfonline.com/doi/abs/10.1080/00401706.1992.10485232), a commonly applied multivariate method in statistical process control (SPC). |
| VAR-T2       | [Vector Autoregression + Hotelling's T-Squared](https://doi.org/10.1016/j.ijpe.2006.07.002). Vector autoregression models the serial correlation in the data, and Hotelling's T2 statistic is then used to detect anomalies in residuals between the predicted and observed values of the data. |
| OC-SVM       | [One-Class Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) |
| LODA         | [Lightweight On-Line Detector of Anomalies](https://doi.org/10.1007/s10994-015-5521-0) |
| EncDec-AD    | [LSTM Encoder-Decoder](https://arxiv.org/abs/1607.00148) |

## Installation

Clone the repository and install requirements in your environment:

```
pip install -r requirements.txt
```

## Usage

See [Jupyter Notebook Sample](examples/artificial_data.ipynb) that displays how to use the models with simulated data.

## License

[Apache License 2.0](LICENSE)

## Contact

Work of Aalto Factory of the Future (https://www.aalto.fi/en/futurefactory), Aalto University

Udayanto Dwi Atmojo (repo maintainer) udayanto(dot)atmojo(at)aalto(dot)fi
Vili Ketonen (author)

[![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/vili-ketonen-75b16267)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/viliket/)
