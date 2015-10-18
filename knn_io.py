""" 
Copyright (C) 2015  Nicola Dileo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>


Module: knn_io.py
--------
"""

from sklearn.datasets.base import Bunch
import numpy as np

def load_dataset(filename):
	raw_data = np.genfromtxt(filename)
	samples = raw_data[:,0:-1]   #getting samples
	targets = raw_data[:,-1]     #getting target attributes
	bun = Bunch(DESCR = filename,
                  data = samples,
		  target = targets)
	return bun



