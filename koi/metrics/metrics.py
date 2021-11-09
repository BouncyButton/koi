import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# should move as trainer method?
from tqdm import tqdm


def generative_negative_error(trainer, N1=2, N2=100):
    model = trainer.model
    X, y = trainer.test.X_original, trainer.test.y_original
    X = torch.stack(X).detach().cpu().numpy()
    y = np.array(y)
    #X, y = torch.tensor(X).detach().cpu().numpy(), torch.tensor(y).detach().cpu().numpy()
    x_dim = trainer.config.x_dim
    device = trainer.device
    latent_size = model.latent_size

    print("training svm model for generative negative error")
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X.reshape(-1, x_dim), y)
    print("training score (svm): ", clf.score(X.reshape(-1, x_dim), y))

    tot_neg = 0

    for i in tqdm(range(N1)):
        z = torch.randn([N2, latent_size]).to(device)
        sampled_x = model.inference(z).cpu().detach().numpy()
        svm_result = clf.predict(sampled_x)
        tot_neg += np.count_nonzero(svm_result == 0)

    print("total negative: ", tot_neg, "{0:.2%}".format(tot_neg / 100000))
    return tot_neg / 100000
