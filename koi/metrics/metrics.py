import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def generative_negative_error(trainer):
    model = trainer.model
    X, y = trainer.test.X_original, trainer.test.y_original
    x_dim = X.shape[1]
    device = trainer.device
    latent_size = model.latent_size

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X.reshape(-1, x_dim), y)

    tot_neg = 0

    for i in range(100):
        z = torch.randn([1000, latent_size]).to(device)
        sampled_x = model.inference(z).cpu().detach().numpy()
        svm_result = clf.predict(sampled_x)
        tot_neg += np.count_nonzero(svm_result == 0)

    print("total negative: ", tot_neg, "{0:.2%}".format(tot_neg / 100000))
    return tot_neg / 100000
