import numpy as np
from monai.data import GridPatchDataset, DataLoader, PatchIter, PatchDataset, Dataset
from monai.transforms import RandShiftIntensity, RandSpatialCropSamples, Compose, RandFlipd
import SimpleITK as sitk
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import Subset
import collections.abc
from monai.transforms import Compose, Randomizable, Transform, apply_transform, RandRotated, RandAffined
import matplotlib.pyplot as plt


class Dataset2hD(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        '''
        :param data: Tuples of img,seg files
        :param transform:
        '''

        # img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][0])),0) # needed?
        # indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][1])),0)# needed?
        # GM = indxs == 2# needed?
        # WM = indxs == 1# needed?
        # seg = np.concatenate( (WM,GM),axis=0) # needed?
        for k in dataurls: # why starting from [1]? -> changed to whole training_cases
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f"{datafolder}/{k}/rawavg.nii")),0) # needed?
            # cseg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f"{datafolder}/{k}/aseg.nii")),0) # needed?
            imga = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0) # why expand with one? for matrix purpose calculating theta/bias?# needed?
            if (imga.shape[2] != 512) or (imga.shape[3] != 512):
                continue
            sega = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[1])), 0)  # why expand again?
            if (sega.shape[2] != 512) or (sega.shape[3] != 512):
                continue
            valid_slices = np.argwhere(np.squeeze(sega).sum(axis=2).sum(axis=1) > 0) # only slices with lesion
            img = imga[:,valid_slices[:,0],:,:] # reduce img to valid slides in dimension [1]
            seg = sega[:,valid_slices[:,0],:,:] # reduce seg to valid slides in dimension [1]
            # GM = indxs == 2
            # WM = indxs == 1


            # cseg = np.concatenate((WM, GM), axis=0) # needed?

            # img = np.concatenate((img, cimg), axis=1) # needed?
            # seg = np.concatenate((seg, cseg), axis=1) # needed?

        # lets just look at AUG for 1 slice
        indexx = 5
        if 0: # what is 0?
            sampleimg = img[:,indexx, :, :]
            sampleseg = seg[:,indexx, :, :]
            for islice in range(img.shape[1]):
                img[:,islice, :, :] = sampleimg
                seg[:,islice, :, :] = sampleseg

        img[img<0] = 0 # defines the window?
        img[img>50] = 50 # defines the window?
        images = {"img": (img-30.0)/5.0, "seg": seg.astype(np.uint8)} # forms a dictionary, why (img-30.0)/5.0 normalization?
        super().__init__( images, transform )

    def __len__(self) -> int:
        return self.data["img"].shape[1]-2

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        # data_i = self.data[index]
        imgblock = self.data["img"][0,index:index+3,:,:] # what does that do?
        segblock = self.data["seg"][:,index+1,:,:]
        data_i = {"img": imgblock, "seg": segblock} # dictionary
        #data_i = self.data[index]

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i



    def __getitem__(self, index: Union[int, slice, Sequence[int]]):  # to analyse just a part of whole data to safe time?
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            #imgblock = np.concatenate( (self.data["img"][index],self.data["img"][index+1],self.data["img"][index+2]))
            #segblock = np.concatenate( (self.data["seg"][index], self.data["seg"][index + 1], self.data["seg"][index + 2]))
            #return {"img":imgblock,"seg":segblock}
            return Subset(dataset=self, indices=index)
        return self._transform(index)


