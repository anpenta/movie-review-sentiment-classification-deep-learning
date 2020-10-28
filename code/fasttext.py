# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# FastText Module
# FastText neural network model.

import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):

  def __init__(self, input_dimension_size, embedding_dimension_size, output_dimension_size):
    super().__init__()
    self._embedding = nn.Embedding(input_dimension_size, embedding_dimension_size)
    self._output = nn.Linear(embedding_dimension_size, output_dimension_size)

  def forward(self, x):
    x = self._embedding(x)
    x = x.permute(1, 0, 2)
    x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
    x = self._output(x)
    return x

