#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss_functions.TopK_loss import TopKLoss


class nnUNetTrainer_Loss_TopK10(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = TopKLoss(k=10)


nnUNetTrainer_Loss_TopK10_copy1 = nnUNetTrainer_Loss_TopK10
nnUNetTrainer_Loss_TopK10_copy2 = nnUNetTrainer_Loss_TopK10
nnUNetTrainer_Loss_TopK10_copy3 = nnUNetTrainer_Loss_TopK10
nnUNetTrainer_Loss_TopK10_copy4 = nnUNetTrainer_Loss_TopK10
