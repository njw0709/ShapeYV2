import h5py
from typing import List
import numpy as np
from .. import custom_dataclasses as cd
from .. import utils


class HDFProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_keys(hdfstore: h5py.File) -> List[str]:
        return list(hdfstore.keys())

    @staticmethod
    def get_whole_coormat(hdfstore: h5py.File, key: str) -> cd.WholeCorrMat:
        return cd.WholeCorrMat(
            dims=hdfstore[key].shape,
            corrmat=hdfstore[key][:],
        )

    @staticmethod
    def get_partial_coormat(
        hdfstore: h5py.File, key: str, coords: cd.Coordinates
    ) -> cd.PartialCorrMat:
        if isinstance(coords.x, tuple):
            if isinstance(coords.y, tuple):
                coormat_partial_np = hdfstore[key][
                    coords.x[0] : coords.x[1], coords.y[0] : coords.y[1]
                ]
            else:
                coormat_partial_np = hdfstore[key][coords.x[0] : coords.x[1], coords.y]
        else:
            if isinstance(coords.y, tuple):
                coormat_partial_np = hdfstore[key][coords.x, coords.y[0] : coords.y[1]]
            else:
                coormat_partial_np = hdfstore[key][coords.x, coords.y]
        return cd.PartialCorrMat(
            dims=coormat_partial_np.shape,
            corrmat=coormat_partial_np,
            coordinates=coords,
        )

    @staticmethod
    def get_imgnames(hdfstore: h5py.File, imgname_key: str) -> List[str]:
        imgnames = hdfstore[imgname_key][:].astype("U")
        axes_of_interest = utils.AXES_OF_INTEREST
        objnames = np.unique(np.array([c.split("-")[0] for c in imgnames]))
        return cd.ImageNames(
            imgnames=imgnames, objnames=objnames, axes_of_interest=axes_of_interest
        )
