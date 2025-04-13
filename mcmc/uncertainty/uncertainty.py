import os
import pickle as pkl
import warnings

import numpy as np
import torch

from .gmm import GaussianMixture
from .prediction import get_prediction, get_residual, get_system_val


class Uncertainty:
    def __init__(
        self,
        order: str,
        calibrate: bool,
        cp_alpha: None | float = 0.05,
        min_uncertainty: float = None,
        *args,
        **kwargs,
    ):
        assert order in [
            "atomic",
            "system_sum",
            "system_mean",
            "system_max",
            "system_min",
            "system_mean_squared",
            "system_root_mean_squared",
        ], f"{order} not implemented"
        self.order = order
        self.calibrate = calibrate
        self.umin = min_uncertainty
        self.cp_alpha = cp_alpha

        if self.calibrate:
            assert cp_alpha is not None, "cp_alpha must be specified for calibration"

            self.CP = ConformalPrediction(alpha=cp_alpha)

    def __call__(self, *args, **kwargs):
        return self.get_uncertainty(*args, **kwargs)

    def set_min_uncertainty(self, uncertainty, force=False):
        if self.umin is None:
            self.umin = uncertainty
        elif force:
            warnings.warn(f"Uncertainty: min_uncertainty already set to {self.umin}. Overwriting.")
            self.umin = uncertainty
        else:
            raise Exception(f"Uncertainty: min_uncertainty already set to {self.umin}")

    def scale_to_min_uncertainty(
        self, uncertainty: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Scale the uncertainty to the minimum value."""
        if self.umin is not None:
            if self.order not in ["system_mean_squared"]:
                uncertainty = uncertainty - self.umin
            else:
                uncertainty = uncertainty - self.umin**2

        return uncertainty

    def fit_conformal_prediction(
        self,
        residuals_calib: np.ndarray | torch.Tensor,
        heuristic_uncertainty_calib: np.ndarray | torch.Tensor,
    ) -> None:
        """Fit the Conformal Prediction model to the calibration data."""
        self.CP.fit(residuals_calib, heuristic_uncertainty_calib)

    def calibrate_uncertainty(
        self, uncertainty: np.ndarray | torch.Tensor, *args, **kwargs
    ) -> np.ndarray | torch.Tensor:
        """Calibrate the uncertainty using Conformal Prediction."""
        if self.CP.qhat is None:
            raise Exception("Uncertainty: ConformalPrediction not fitted.")

        cp_uncertainty, qhat = self.CP.predict(uncertainty)

        return cp_uncertainty

    def get_uncertainty(self, results, *args, **kwargs):
        return NotImplementedError

    def get_input_params(self):
        return NotImplementedError

    def save(self, path):
        unc_type, inputs = self.get_input_params()

        pkl.dump({"uncertainty_type": unc_type, "unc_params": inputs}, open(path, "wb"))

    @classmethod
    def load(cls, path):
        loaded_info_dict = pkl.load(open(path, "rb"))
        if loaded_info_dict["unc_params"]["calibrate"]:
            qhat = loaded_info_dict["unc_params"].pop("qhat")
            unc_class = UNC_DICT[loaded_info_dict.pop("uncertainty_type")](
                **loaded_info_dict["unc_params"]
            )
            unc_class.CP.qhat = qhat
        else:
            unc_class = UNC_DICT[loaded_info_dict.pop("uncertainty_type")](
                **loaded_info_dict["unc_params"]
            )
            unc_class.CP.qhat

        return unc_class


class ConformalPrediction:
    """Copied from https://github.com/ulissigroup/amptorch
    Performs quantile regression on score functions to obtain the estimated qhat
        on calibration data and apply to test data during prediction.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.qhat = None

    def fit(
        self,
        residuals_calib: np.ndarray | torch.Tensor,
        heuristic_uncertainty_calib: np.ndarray | torch.Tensor,
    ) -> None:
        # score function
        scores = abs(residuals_calib / heuristic_uncertainty_calib)
        scores = np.array(scores)

        n = len(residuals_calib)
        qhat = torch.quantile(torch.from_numpy(scores), np.ceil((n + 1) * (1 - self.alpha)) / n)
        qhat_value = np.float64(qhat.numpy()).item()
        self.qhat = qhat_value

    def predict(
        self, heuristic_uncertainty_test: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, float]:
        cp_uncertainty_test = heuristic_uncertainty_test * self.qhat
        return cp_uncertainty_test, self.qhat


class EnsembleUncertainty(Uncertainty):
    """Ensemble uncertainty estimation using the variance or standard deviation of the
    predictions from the model ensemble.
    """

    def __init__(
        self,
        quantity: str,
        order: str,
        std_or_var: str = "var",
        min_uncertainty: float | None = None,
        # orig_unit: Union[str, None] = None,
        # targ_unit: Union[str, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            min_uncertainty=min_uncertainty,
            calibrate=False,
            *args,
            **kwargs,
        )
        assert std_or_var in ["std", "var"], f"{std_or_var} not implemented"
        self.q = quantity
        # self.orig_unit = orig_unit
        # self.targ_unit = targ_unit
        self.std_or_var = std_or_var

    def get_energy_uncertainty(
        self,
        results: dict,
    ):
        """Get the uncertainty for the energy."""
        # if self.orig_unit is not None and self.targ_unit is not None:
        #     results["energy"] = self.convert_units(
        #         results["energy"], orig_unit=self.orig_unit, targ_unit=self.targ_unit
        #     )

        if self.std_or_var == "std":
            val = results["energy" + "_std"]
        elif self.std_or_var == "var":
            val = results["energy" + "_var"] ** 2

        return val

    def get_forces_uncertainty(
        self,
        results: dict,
        num_atoms: list,
    ):
        # if self.orig_unit is not None and self.targ_unit is not None:
        #     # results["forces"] = self.convert_units(
        #     #     results["forces"], orig_unit=self.orig_unit, targ_unit=self.targ_unit
        #     # )
        #     results["forces_std"] = self.convert_units(
        #         results["forces_std"], orig_unit=self.orig_unit, targ_unit=self.targ_unit
        #     )
        if self.std_or_var == "std":
            val = torch.norm(results["forces_std"], dim=-1)
        elif self.std_or_var == "var":
            val = torch.norm(results["forces_std"] ** 2, dim=-1)
        if "system" in self.order:
            system_forces = get_system_val(val, num_atoms, self.order)
            return system_forces
        else:
            return val

    def get_uncertainty(self, results: dict, num_atoms: list = None, *args, **kwargs):
        if self.q == "energy_std":
            val = self.get_energy_uncertainty(results=results)
        elif self.q in ["energy_grad_std", "forces_std"]:
            val = self.get_forces_uncertainty(
                results=results,
                num_atoms=num_atoms,
            )
        else:
            raise TypeError(f"{self.q} not yet implemented")
        if self.umin is not None:
            val = self.scale_to_min_uncertainty(val)

        return val

    def get_input_params(self):
        unc_type = "ensemble"
        inputs = {
            "quantity": self.q,
            "order": self.order,
            "std_or_var": self.std_or_var,
            "min_uncertainty": self.umin,
        }
        return unc_type, inputs


class GMMUncertainty(Uncertainty):
    """Gaussian Mixture Model (GMM) based uncertainty estimation."""

    def __init__(
        self,
        train_embed_key: str = "embedding",
        test_embed_key: str = "embedding",
        n_clusters: int = 5,
        order: str = "atomic",
        covariance_type: str = "full",
        tol: float = 1e-3,
        max_iter: int = 100000,
        n_init: int = 1,
        init_params: str = "kmeans",
        verbose: int = 0,
        device: str = "cuda",
        calibrate: bool = False,
        cp_alpha: float | None = None,
        min_uncertainty: float | None = None,
        gmm_path: str | None = None,
        gm_model: GaussianMixture | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            calibrate=calibrate,
            cp_alpha=cp_alpha,
            min_uncertainty=min_uncertainty,
            *args,
            **kwargs,
        )
        self.train_key = train_embed_key
        self.test_key = test_embed_key
        self.n = n_clusters
        self.covar_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.verbose = verbose
        self.device = device

        self.gmm_path = gmm_path
        if gmm_path is not None and os.path.exists(gmm_path):
            import pickle

            with open(gmm_path, "rb") as f:
                self.gm_model = pickle.load(f)
            # Set the GMM parameters if the model is loaded
            self._set_gmm_params()
        elif gm_model is not None:
            self.gm_model = gm_model
            self._set_gmm_params()
        else:
            print(f"gm_model {gmm_path} does not exist")

    def fit_gmm(self, Xtrain: torch.Tensor) -> None:
        """Fit the GMM model to the embedding of training data."""
        self.Xtrain = Xtrain
        self.gm_model = GaussianMixture(
            n_components=self.n,
            covariance_type=self.covar_type,
            tol=self.tol,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            verbose=self.verbose,
        )
        self.gm_model.fit(self.Xtrain.squeeze().cpu().numpy())

        # Save the fitted GMM model if gmm_path is specified
        if hasattr(self, "gmm_path") and not os.path.exists(self.gmm_path):
            self.gm_model.save(self.gmm_path)
            print(f"Saved fitted GMM model to {self.gmm_path}")

        # Set the GMM parameters
        self._set_gmm_params()

    def is_fitted(self) -> bool:
        """Check if the GMM model is fitted."""
        return getattr(self, "gm_model", None) is not None

    def _check_tensor(
        self,
        X: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Check if the input is a tensor and convert to torch.Tensor if not."""
        if isinstance(X, list) and isinstance(X[0], (np.ndarray, torch.Tensor)):
            X = torch.cat([torch.tensor(x) for x in X])
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        X = X.squeeze().double().to(self.device)

        return X

    def _set_gmm_params(self) -> None:
        """Get the means, precisions_cholesky, and weights from the GMM model."""
        if self.gm_model is None:
            raise Exception("GMMUncertainty: GMM does not exist/is not fitted")

        self.means = self._check_tensor(self.gm_model.means_)
        self.precisions_cholesky = self._check_tensor(self.gm_model.precisions_cholesky_)
        self.weights = self._check_tensor(self.gm_model.weights_)

    def estimate_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate the log probability of the given embedding."""
        X = self._check_tensor(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        n_samples, n_features = X.shape
        n_clusters, _ = self.means.shape

        log_det = torch.sum(
            torch.log(self.precisions_cholesky.reshape(n_clusters, -1)[:, :: n_features + 1]),
            dim=1,
        )

        log_prob = torch.empty((n_samples, n_clusters)).to(X.device)
        for k, (mu, prec_chol) in enumerate(
            zip(self.means, self.precisions_cholesky, strict=False)
        ):
            y = torch.matmul(X, prec_chol) - (mu.reshape(1, -1) @ prec_chol).squeeze()
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        log2pi = torch.log(torch.tensor([2 * torch.pi])).to(X.device)
        return -0.5 * (n_features * log2pi + log_prob) + log_det

    def estimate_weighted_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate the weighted log probability of the given embedding."""
        log_prob = self.estimate_log_prob(X)
        log_weights = torch.log(self.weights)
        weighted_log_prob = log_prob + log_weights

        return weighted_log_prob

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Log likelihood of the embedding under the GMM model."""
        weighted_log_prob = self.estimate_weighted_log_prob(X)

        weighted_log_prob_max = weighted_log_prob.max(dim=1).values
        # logsumexp is numerically unstable for big arguments
        # below, the calculation below makes it stable
        # log(sum_i(a_i)) = log(exp(a_max) * sum_i(exp(a_i - a_max))) = a_max + log(sum_i(exp(a_i - a_max)))
        wlp_stable = weighted_log_prob - weighted_log_prob_max.reshape(-1, 1)
        logsumexp = weighted_log_prob_max + torch.log(torch.sum(torch.exp(wlp_stable), dim=1))

        return logsumexp

    def probability(self, X: torch.Tensor) -> torch.Tensor:
        """Probability of the embedding under the GMM model."""
        logP = self.log_likelihood(X)

        return torch.exp(logP)

    def negative_log_likelihood(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log likelihood of the embedding under the GMM model."""
        logP = self.log_likelihood(X)

        return -logP

    def get_uncertainty(
        self,
        results: dict,
        num_atoms: list[int] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Get the uncertainty from the GMM model for the test embedding.

        Args:
            results (dict): Dictionary containing the results from the model.
            num_atoms (Union[List[int], None], optional): Number of atoms in each system. Defaults to None.

        Returns:
            torch.Tensor: Uncertainty from the GMM model.
        """
        test_embedding = self._check_tensor(results[self.test_key])

        if self.is_fitted() is False:
            train_embedding = self._check_tensor(results[self.train_key])
            self.fit_gmm(train_embedding)

        uncertainty = self.negative_log_likelihood(test_embedding)
        if "system" in self.order:
            uncertainty = get_system_val(uncertainty, num_atoms, self.order)
        if self.umin is not None:
            uncertainty = self.scale_to_min_uncertainty(uncertainty)
        if self.calibrate:
            uncertainty = self.calibrate_uncertainty(uncertainty)

        return uncertainty

    def get_input_params(self):
        inputs = {
            "train_embed_key": self.train_key,
            "test_embed_key": self.test_key,
            "n_clusters": self.n,
            "order": self.order,
            "covariance_type": self.covar_type,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "verbose": self.verbose,
            "calibrate": self.calibrate,
            "cp_alpha": self.cp_alpha,
            "min_uncertainty": self.umin,
        }
        if self.gm_model is not None:
            inputs.update({"gm_model": self.gm_model})
        else:
            inputs.update({"gmm_path": self.gmm_path})
        if self.calibrate:
            inputs.update({"qhat": self.CP.qhat})
        return "gmm", inputs


UNC_DICT = {
    "ensemble": EnsembleUncertainty,
    "gmm": GMMUncertainty,
    # "evidential": EvidentialUncertainty,
    # "mve": MVEUncertainty,
}


def get_unc_class(model, info_dict):
    device = info_dict["device"]
    model.eval()

    unc_class = UNC_DICT[info_dict["uncertainty_type"]](**info_dict["uncertainty_params"])
    # turn off calibration for now in CP for initial fittings
    unc_class.calibrate = False

    if info_dict.get("uncertainty_type") == "gmm":
        # if the unc_class already has a gm_model, then we don't need
        # to refit it
        if unc_class.is_fitted() is False:
            print("COLVAR: Doing train prediction")
            _, train_predicted = get_prediction(
                model=model,
                dset=info_dict["train_dset"],
                batch_size=info_dict["batch_size"],
                device=device,
            )

            train_embedding = train_predicted["embedding"].detach().cpu().squeeze()

            print("COLVAR: Fitting GMM")
            unc_class.fit_gmm(train_embedding)
    calibrate = info_dict["uncertainty_params"].get("calibrate", False)
    if calibrate:
        print("COLVAR: Fitting ConformalPrediction")
        calib_target, calib_predicted = get_prediction(
            model=model,
            dset=info_dict["calib_dset"],
            batch_size=info_dict["batch_size"],
            device=device,
        )
        # calib_predicted["embedding"] = calib_predicted["embedding"][0]
        calib_uncertainty = (
            unc_class(
                results=calib_predicted,
                num_atoms=calib_predicted["num_atoms"],
                device=device,
            )
            .detach()
            .cpu()
        )

        # set minimum uncertainty to scale to
        umin = calib_uncertainty.min().item()
        unc_class.set_min_uncertainty(umin)
        print(f"COLVAR: Setting min_uncertainty to {umin}")

        calib_res = (
            get_residual(
                targ=calib_target,
                pred=calib_predicted,
                num_atoms=calib_predicted["num_atoms"],
                quantity=info_dict["uncertainty_params"]["quantity"],
                order=info_dict["uncertainty_params"]["order"],
            )
            .detach()
            .cpu()
        )
        unc_class.fit_conformal_prediction(
            calib_res,
            calib_uncertainty,
        )
        # turn on the calibration again
        unc_class.calibrate = True
    return unc_class
