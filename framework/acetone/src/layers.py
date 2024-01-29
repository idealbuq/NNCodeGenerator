"""
 *******************************************************************************
 * ACETONE: Predictable programming framework for ML applications in safety-critical systems
 * Copyright (c) 2022. ONERA
 * This file is part of ACETONE
 *
 * ACETONE is free software ;
 * you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation ;
 * either version 3 of  the License, or (at your option) any later version.
 *
 * ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this program ;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
 ******************************************************************************
"""

from re import M
import numpy as np
from abc import ABC, abstractmethod

class Layers(ABC):
    
    def __init__(self):

        self.idx = 0
        self.size = 0
        self.name = ''
        self.next_layer = [] 
        self.previous_layer = []
        self.globalvars_str = ''
        self.header_str = ''
        self.source_str = ''
      
        super().__init__()

    @abstractmethod
    def feedforward(self):
        pass

    def flatten_array_hybrid(self, array):
        ndim = array.ndim
        array = array.reshape(-1, *array.shape[-(ndim-2):])
        
        flattened_aray = array.flatten(order='F')
        s = '\n        {'
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i])+', '
        s = s[:-2]
        s+='}'
        
        return s

    def count_elements_array(self, array):
        nb_elements = 1
        for dim in np.shape(array) : nb_elements *= dim
        return nb_elements

    def compute_padding(self, in_height, in_width, kernel_h, kernel_w, strides, dilation_rate=1):
        
        # Compute 'same' padding tensorflow

        filter_height = (kernel_h - (kernel_h-1)*(dilation_rate-1))
        filter_width = (kernel_w - (kernel_w-1)*(dilation_rate-1))

        # The total padding applied along the height and width is computed as:

        if (in_height % strides == 0):
            pad_along_height = max(filter_height - strides, 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides), 0)
        if (in_width % strides == 0):
            pad_along_width = max(filter_width - strides, 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_right, pad_left, pad_bottom, pad_top
       
class InputLayer(Layers):

    def __init__(self, idx, size):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Input_layer'

    def write_to_function_source_file(self, source_file):
        
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        output_pre[i] = nn_input[i]; \n    } \n\n')

    def feedforward(self, input):
        
        return input 

class Dense(Layers):

    def __init__(self, idx, size, weights, biases, activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Dense'
        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

    def write_to_function_source_file(self, source_file):

        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        dotproduct = 0;\n')
        source_file.write( '        for (int j = 0; j < ' + str(self.previous_layer[0].size) + '; ++j)\n        {\n')
        source_file.write( '            dotproduct += output_pre[j] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[(j + ' + str(self.previous_layer[0].size) + '*i)];\n        }\n')
        source_file.write( '        dotproduct += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[i];\n')

        a = self.activation_function.write_activation_str(self.local_var)

        source_file.write( '        output_cur[i] = '+ a +';\n    }\n\n')

    def feedforward(self, input):

        input = input.reshape((self.previous_layer[0]).size) 

        return self.activation_function.compute((np.dot(input, self.weights) + self.biases))
     
class Conv2D(Layers):
    
    def __init__(self, idx, conv_algorithm, data_format, size, padding, strides, kernel_h, kernel_w, dilation_rate, nb_filters, input_shape, output_shape, weights, biases, activation_function):
        
        super().__init__()
        self.conv_algorithm = conv_algorithm
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = 'Conv2D'
        self.padding = padding
        self.strides = strides
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'sum'

        if self.data_format == 'channels_first':
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]
            self.weights_py_inf = np.moveaxis(self.weights, 2, 0)


        elif self.data_format == 'channels_last':
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            self.input_channels = input_shape[3]
            self.output_height = output_shape[1]
            self.output_width = output_shape[2]


        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        if self.padding == 'same':
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.input_height, self.input_width, self.kernel_h, self.kernel_w, self.strides, self.dilation_rate)
        else:
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = 0, 0, 0, 0

    @abstractmethod
    def write_to_function_source_file(self, source_file):
        pass

    def feedforward(self, input):
        # Treat input
        if(self.data_format == 'channels_first'):
            input = input.reshape(self.input_channels, self.input_height, self.input_width)
        
        elif(self.data_format == 'channels_last'):
            input = input.reshape(self.input_height, self.input_width, self.input_channels)
            input= np.transpose(input,(2,0,1))
        
        # General both for CHW and HWC
        output = np.zeros((self.nb_filters, self.output_height, self.output_width))
        
        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input

        for f in range(self.nb_filters):
            for i in range(self.output_height):
                for j in range(self.output_width): 
                    output[f,i,j]=np.sum(input_padded[:, i*self.strides:i*self.strides+self.kernel_h, j*self.strides:j*self.strides+self.kernel_w] 
                                        * self.weights_py_inf[:,:,:,f]) + self.biases[f]
        
        # If HWC, transpose output
        if(self.data_format == 'channels_last'):
            output= np.transpose(output,(1,2,0))

        return self.activation_function.compute(output)

class Conv2D_6loops(Conv2D):
    """Implements Conv2D using the six-loops algorithm (direc conv)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def write_to_function_source_file(self, source_file):
         
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.nb_filters) + '; ++f)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')
        source_file.write('                sum = 0;\n')
        source_file.write('                for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n                {\n')
        source_file.write('                    for (int m = 0; m < '+str(self.kernel_h)+'; ++m)\n                    {\n')
        source_file.write('                        for (int n = 0; n < '+str(self.kernel_w)+'; ++n)\n                        {\n')
        source_file.write('                            int ii = i*'+str(self.strides)+' + m*'+str(self.dilation_rate)+' - '+str(self.pad_left)+';\n')
        source_file.write('                            int jj = j*'+str(self.strides)+' + n*'+str(self.dilation_rate)+' - '+str(self.pad_top)+';\n\n')
        source_file.write('                            if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                            {\n')

        source_file.write('                                sum += output_pre[jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[n + '+str(self.kernel_w)+'*(m + '+str(self.kernel_h)+'*(c + '+str(self.input_channels)+'*f))];\n')
        source_file.write('                            }\n                        }\n                    }\n                }\n')                                            
        source_file.write('                sum += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[f];\n'            )
        
        a = self.activation_function.write_activation_str(self.local_var)
                    
        source_file.write('                output_cur[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)] = '+ a +';\n')
        source_file.write('            }\n        }\n    }\n\n')

class Conv2D_gemm(Conv2D):
    """Implements Conv2D using indirect im2col (or im2row) and GeMM"""
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.patches_height = self.input_channels * self.kernel_h * self.kernel_w
        self.patches_width = self.output_height * self.output_width
        self.patches_size = self.patches_height * self.patches_width

        self.conv_algorithm = self.conv_algorithm[-7:]
        self.algo_gemm_mapping = { 'gemm_nn' : self.write_gemm_nn,
                                   'gemm_nt' : self.write_gemm_nt,
                                   'gemm_tn' : self.write_gemm_tn,
                                   'gemm_tt' : self.write_gemm_tt}

   
    @abstractmethod
    def write_gemm_nn(self, m, n, k, A, B, C):
        pass

    @abstractmethod
    def write_gemm_nt(self, m, n, k, A, B, C):
        pass
    
    @abstractmethod
    def write_gemm_tn(self, m, n, k, A, B, C):
        pass

    @abstractmethod
    def write_gemm_tt(self, m, n, k, A, B, C):
       pass

class Conv2D_indirect_gemm(Conv2D_gemm):
    """Implements Conv2D using indirect im2col (or im2row) and GeMM"""
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_ppatches(self):
        if (self.pad_right or self.pad_left or self.pad_bottom or self.pad_top):
            self.input_h_padded = self.input_height + self.pad_top + self.pad_bottom
            self.input_w_padded = self.input_width + self.pad_left + self.pad_right

            start_idx = np.arange(self.kernel_h)[:,None]*self.input_w_padded + np.arange(self.kernel_w)
            c=self.input_h_padded*self.input_w_padded*np.arange(self.input_channels)
            start_idx=(c[:,None]+start_idx.ravel()).reshape((-1,self.kernel_h,self.kernel_w))
            offset_idx = np.arange(self.output_height, step=self.strides)[:,None]*self.input_w_padded + np.arange(self.output_width, step=self.strides)
            idx_padded_input = (start_idx.ravel()[:,None] + offset_idx.ravel()).flatten()

            idx_of_zeros = []
            j_zeros = np.concatenate((np.arange(self.pad_left), np.arange(self.pad_right)+(self.input_w_padded-self.pad_right)))
            i_zeros = np.concatenate((np.arange(self.pad_top), np.arange(self.pad_bottom)+(self.input_h_padded-self.pad_bottom)))
            for c in range(self.input_channels):
                for i in range(self.input_h_padded):
                    for j in range(self.input_w_padded):
                        if (np.isin(i, i_zeros) or np.isin(j, j_zeros)):
                            idx_of_zeros.append(j + self.input_w_padded*(i+self.input_h_padded*c))
            
            idx_padded_input = np.where(np.isin(idx_padded_input, idx_of_zeros), np.nan, idx_padded_input)
            _, idx_padded_input = np.unique(idx_padded_input, return_inverse=True) 
            self.ppatches=np.where(idx_padded_input==self.input_shape, np.nan, idx_padded_input)

        else:
            start_idx = np.arange(self.kernel_h)[:,None]*self.input_width + np.arange(self.kernel_w)
            c=self.input_height*self.input_width*np.arange(self.input_channels)
            start_idx=(c[:,None]+start_idx.ravel()).reshape((-1,self.kernel_h,self.kernel_w))
            offset_idx = np.arange(self.output_height, step=self.strides)[:,None]*self.input_width + np.arange(self.output_width, step=self.strides)
            self.ppatches = (start_idx.ravel()[:,None] + offset_idx.ravel()).flatten()
            
        if ('gemm_nt' or 'gemm_tt') in self.conv_algorithm:
            self.ppatches = self.ppatches.reshape((self.patches_height, self.patches_width)).transpose().flatten()  
  

        s = '\n        {'
        for i in range(len(self.ppatches)):
            if np.isnan(self.ppatches[i]):
                s += '&zero, '
            else:
                s += '&output_pre[' + str(int(self.ppatches[i])) + '], '

        s=s[:-2]
        s+='}'

        return s

    def write_gemm_nn(self, m, n, k, A, B, C, ):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = n
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('output')

        s = '    // gemm_nn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '        for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '            float register weight = '+str(self.A)+'[i*'+str(self.ldA)+'+p];\n'
        s+= '            for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '                '+str(self.C)+'[i*'+str(self.ldC)+' + j] += weight * *('+str(self.B)+'[p*'+str(self.ldB)+' + j]);\n'
        s+= '            }\n'
        s+= '        }\n'
        s+= '        for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '            float register output = ' +str(self.C)+'[i*'+str(self.ldC)+' + j];\n'
        s+= '            output += biases_' + self.name + '_' + str('{:02d}'.format(self.idx))+'[i];\n'
        s+= '            '+str(self.C)+'[i*'+str(self.ldC)+' + j] = '+a+';\n'
        s+= '        }\n'
        s+= '    }\n\n'
        
        return s
    
    def write_gemm_nt(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = k
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('output')

        s = '    // gemm_nt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register output = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               output += '+str(self.A)+'[i*'+str(self.ldA)+'+p] * *('+str(self.B)+'[j*'+str(self.ldB)+' + p]);\n'
        s+= '           }\n'
        s+= '           output += biases_'+ self.name + '_' + str("{:02d}".format(self.idx))+'[i];\n'
        s+= '           '+str(self.C)+'[i*'+str(self.ldC)+' + j] = '+a+';\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tn(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = n
        self.C = C
        self.ldC = n

        s = '    // gemm_tn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '           float register weight = '+str(self.A)+'[p*'+str(self.ldA)+'+i];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               '+str(self.C)+'[i*'+str(self.ldC)+' + j] += weight * *('+str(self.B)+'[p*'+str(self.ldB)+' + j]);\n'
        s+= '           }\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tt(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = k
        self.C = C
        self.ldC = n

        s = '    // gemm_tt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register sum = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               sum += '+str(self.A)+'[p*'+str(self.ldA)+'+i] * *('+str(self.B)+'[j*'+str(self.ldB)+' + p]);\n'
        s+= '           }\n'
        s+= '           '+str(self.C)+'[i*'+str(self.ldC)+' + j] += sum;\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_to_function_source_file(self, source_file):
        
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')   
        gemm_code = self.algo_gemm_mapping[self.conv_algorithm](self.nb_filters, self.patches_width, self.patches_height, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), 'ppatches_' + self.name + '_' + str("{:02d}".format(self.idx)), "output_cur")
        source_file.write(gemm_code)
        
        return 0

class Conv2D_std_gemm(Conv2D_gemm):
    """Implements Conv2D using im2col (or im2row) and GeMM"""
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.algo_patch_building_mapping = { 'gemm_nn' : self.write_im2col,
                                             'gemm_nt' : self.write_im2row,
                                             'gemm_tn' : self.write_im2col,
                                             'gemm_tt' : self.write_im2row}

    def write_im2col(self):
        s = '    // im2col\n'
        s+= '    for (int i = 0; i < '+str(self.patches_height)+'; ++i) {\n\n'
        s+= '        int i_offset = (i / '+str(self.kernel_w)+') % '+str(self.kernel_h)+';\n'
        s+= '        int j_offset = i % '+str(self.kernel_w)+';\n'
        s+= '        int c_offset = i / '+str(self.kernel_h)+' / '+str(self.kernel_w)+';\n\n'
        s+= '        for (int h = 0; h < '+str(self.output_height)+'; ++h) {\n'
        s+= '            for (int w = 0; w < '+str(self.output_width)+'; ++w) {\n\n'
        s+= '                int ii = h * '+str(self.strides)+' - '+str(self.pad_top)+' + i_offset; \n'
        s+= '                int jj = w * '+str(self.strides)+' - '+str(self.pad_left)+' + j_offset;\n\n'
        s+= '                int j = h*'+str(self.output_width)+' + w;\n'
        s+= '                if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n'
        s+= '                    output_cur[i*'+str(self.patches_width)+' + j] = output_pre[(c_offset*'+str(self.input_height)+' + ii)*'+str(self.input_width)+' + jj];\n'
        s+= '                else\n'
        s+= '                    output_cur[i*'+str(self.patches_width)+' + j] = 0;\n'
        s+= '            }\n'
        s+= '        }\n'
        s+= '    }\n'
        s+= '    \n\n'
        
        return s

    def write_im2row(self):
        s = '    // im2row\n'
        s+= '    for (int i = 0; i < '+str(self.patches_height)+'; ++i) {\n\n'
        s+= '        int i_offset = (i / '+str(self.kernel_w)+') % '+str(self.kernel_h)+';\n'
        s+= '        int j_offset = i % '+str(self.kernel_w)+';\n'
        s+= '        int c_offset = i / '+str(self.kernel_h)+' / '+str(self.kernel_w)+';\n\n'
        s+= '        for (int h = 0; h < '+str(self.output_height)+'; ++h) {\n'
        s+= '            for (int w = 0; w < '+str(self.output_width)+'; ++w) {\n\n'
        s+= '                int ii = h * '+str(self.strides)+' - '+str(self.pad_top)+' + i_offset; \n'
        s+= '                int jj = w * '+str(self.strides)+' - '+str(self.pad_left)+' + j_offset;\n\n'
        s+= '                int j = w*'+str(self.output_height)+' + h;\n'
        s+= '                if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n'
        s+= '                    output_cur[j*'+str(self.patches_height)+' + i] = output_pre[(c_offset*'+str(self.input_height)+' + ii)*'+str(self.input_width)+' + jj];\n'
        s+= '                else\n'
        s+= '                    output_cur[j*'+str(self.patches_height)+' + i] = 0;\n'
        s+= '            }\n'
        s+= '        }\n'
        s+= '    }\n'
        s+= '    \n\n'
        
        return s

    def write_gemm_nn(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = n
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('output')
        s = '    // gemm_nn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '           float register weight = '+str(self.A)+'[i*'+str(self.ldA)+'+p];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               '+str(self.C)+'[i*'+str(self.ldC)+' + j] += weight * '+str(self.B)+'[p*'+str(self.ldB)+' + j];\n'
        s+= '           }\n'
        s+= '       }\n'
        s+= '        for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '            float register output = ' +str(self.C)+'[i*'+str(self.ldC)+' + j];\n'
        s+= '            output += biases_' + self.name + '_' + str('{:02d}'.format(self.idx))+'[i];\n'
        s+= '            '+str(self.C)+'[i*'+str(self.ldC)+' + j] = '+a+';\n'
        s+= '        }\n'
        s+= '    }\n\n'        
        return s
    
    def write_gemm_nt(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = k
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('output')

        s = '    // gemm_nt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register output = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               output += '+str(self.A)+'[i*'+str(self.ldA)+'+p] * '+str(self.B)+'[j*'+str(self.ldB)+' + p];\n'
        s+= '           }\n'
        s+= '           output += biases_'+ self.name + '_' + str("{:02d}".format(self.idx))+'[i];\n'
        s+= '           '+str(self.C)+'[i*'+str(self.ldC)+' + j] += '+a+';\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tn(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = n
        self.C = C
        self.ldC = n

        s = '    // gemm_tn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '           float register weight = '+str(self.A)+'[p*'+str(self.ldA)+'+i];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               '+str(self.C)+'[i*'+str(self.ldC)+' + j] += weight * '+str(self.B)+'[p*'+str(self.ldB)+' + j];\n'
        s+= '           }\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tt(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = k
        self.C = C
        self.ldC = n

        s = '    // gemm_tt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register sum = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               sum += '+str(self.A)+'[p*'+str(self.ldA)+'+i] * '+str(self.B)+'[j*'+str(self.ldB)+' + p];\n'
        s+= '           }\n'
        s+= '           '+str(self.C)+'[i*'+str(self.ldC)+' + j] += sum;\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        patch_building_code = self.algo_patch_building_mapping[self.conv_algorithm]()
        source_file.write(patch_building_code)
        source_file.write('    for (int k = 0; k < '+str(self.patches_height*self.patches_width)+'; ++k){\n        output_pre[k] = output_cur[k];\n        output_cur[k] = 0;\n    }\n')
        gemm_code = self.algo_gemm_mapping[self.conv_algorithm](self.nb_filters, self.patches_width, self.patches_height, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), "output_pre", "output_cur")
        source_file.write(gemm_code)

class Pooling2D(Layers):
    def __init__(self, idx, data_format, size, padding, strides, pool_size, input_shape, output_shape, **kwargs):
        
        super().__init__()
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = ''
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

        if self.data_format == 'channels_first':
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]

        elif self.data_format == 'channels_last':
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            self.input_channels = input_shape[3]
            self.output_height = output_shape[1]
            self.output_width = output_shape[2]

        self.pooling_funtion = ''
        self.local_var = ''
        self.local_var_2 = ''
        self.output_var = ''

        if self.padding == 'same':
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.input_height, self.input_width, self.pool_size, self.strides)
        else:
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = 0, 0, 0, 0

    def generate_output_str(self, index, output):
        
        return '    '+output+'['+index+'] = '+ self.output_var +';\n\n'

    @abstractmethod    
    def specific_function(self, index, input_of_layer):
        pass

    def write_to_function_source_file(self, source_file):
 
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')

        source_file.write('            ' + self.update_local_vars())

        source_file.write('                for (int m = 0; m < '+str(self.pool_size)+'; ++m)\n                {\n')
        source_file.write('                    for (int n = 0; n < '+str(self.pool_size)+'; ++n)\n                    {\n')
        source_file.write('                        int ii = i*'+str(self.strides)+' + m - '+str(self.pad_left)+';\n')
        source_file.write('                        int jj = j*'+str(self.strides)+' + n - '+str(self.pad_top)+';\n\n')
        source_file.write('                        if (ii >= 0 && ii < '+str( self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                        {\n')

        source_file.write(self.specific_function('jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)', 'output_pre'))

        source_file.write('                        }\n                    }\n                }\n')
        source_file.write('            ' + self.generate_output_str('j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*c)', 'output_cur'))
        source_file.write('            }\n        }\n    }\n\n')
      
    def feedforward(self, input):
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        output = np.zeros((self.input_channels, self.output_height, self.output_width))
                
        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input

        for c in range(self.input_channels):
            for i in range(self.output_height):
                for j in range(self.output_width): 
                    output[c,i,j]= self.pooling_function((input_padded[c, i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size]))
        return output

class AveragePooling2D(Pooling2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.name = 'AveragePooling2D'
        self.pooling_function = np.mean
        self.local_var = 'sum'
        self.local_var_2 = 'count'
        self.output_var = self.local_var + '/' + self.local_var_2

    def declare_local_vars(self, data_type):
        
        s = '    '+ data_type + ' '+ self.local_var +';\n'
        s += '    int '+ self.local_var_2 + ';\n\n'

        return s

    def update_local_vars(self):

        s = '    '+ self.local_var + ' = 0; '+ self.local_var_2 + ' = 0;\n'
  
        return s

    def specific_function(self, index, input_of_layer):
        # Computes the average in this subclass AveragePooling2D 
            
        s = '                            '+self.local_var+' += '+input_of_layer+'['+index+'];\n'
        s += '                            '+self.local_var_2+' ++;\n'
        
        return s

class MaxPooling2D(Pooling2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.name = 'MaxPooling2D'
        self.pooling_function = np.amax
        self.local_var = 'max'
        self.output_var = self.local_var

    def declare_local_vars(self, data_type):
        
        s = '    '+ data_type + ' '+ self.local_var +';\n\n'

        return s

    def update_local_vars(self):
        
        s = '    '+ self.local_var +' = -INFINITY;\n'

        return s

    def specific_function(self, index, input_of_layer):
        # Computes the average in this subclass AveragePooling2D 

        s = '                            if ('+input_of_layer+'['+index+'] > '+self.local_var+')\n'
        s += '                                '+self.local_var+' = '+input_of_layer+'['+index+'];\n'

        return s

class Softmax(Layers):

    def __init__(self, idx, size):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Softmax'

    def write_to_function_source_file(self, source_file):
        
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    sum = 0;\n\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; ++i)\n')
        source_file.write('        sum += exp(output_pre[i]);\n\n')
        source_file.write('    for (int j = 0; j < ' + str(self.size) + '; ++j)\n')
        source_file.write('        output_cur[j] = exp(output_pre[j])/sum;\n\n')

    def feedforward(self, input):
        
        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp)

        return output
