from dotmap import DotMap


class GenerativeModel:
    """
    Abstract base class for any generative model implemented by koi.
    Gracefully provided by github.com/makgyver and edited
    """
    def __init__(self, x_dim=2, **kwargs):
        self.config = DotMap()
        self.x_dim = x_dim
        self.kwargs = kwargs

    def save_model(self, filepath, *args, **kwargs):
        r"""Save the model to file.
        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file to save the model.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.
        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemented in the sub-class.
        """
        raise NotImplementedError()

    def load_model(self, filepath, *args, **kwargs):
        r"""Load the model from file.
        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.
        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemented in the sub-class.
        """
        raise NotImplementedError()

    def is_cuda(self):
        raise NotImplementedError
        # TODO next(self.model.parameters()).is_cuda
