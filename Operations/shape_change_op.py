import torch

a = torch.zeros(3,5)

a[0, :] = 1
a[1, :] = 2
a[2, :] = 3

print('a: ',a)
print('shape of a as [seq_len, seq_dim, batch]: ',a.shape) # [seq_len, token_dim, batch]

# Purpose is to make the shape as [token_dim, seq_len, batch]
print('----------------------------------------------------------------------------------')
print('----Requirement is to make the shape of a as [token_dim, seq_len, batch]------------')
print('----------------------------------------------------------------------------------')

print('')
print('How Reshape changes the requirement')
b = a.reshape(5,3)
print('b: ',b)
print('shape of b as [seq_len, token_dim, batch]: ',b.shape)

print('')
print('How Transpose changes the requirement')
c = a.transpose(1,0)
print('c: ',c)
print('shape of c as [seq_len, token_dim, batch]: ',c.shape)


print('')
# More generalized as per the transpose
print('How Permute changes the requirement')
d = a.unsqueeze(0)
print('d shape: ',d.shape)
d = d.permute(2,0,1)
print('d: ',d)
print('shape of d as [token_dim, batch, seq_len]: ',d.shape)


# In pytorch tensor is stored as contiguous list in memory
# Verify is using a.is_contiguous()

print('Is contiguous? ',a.is_contiguous())

# Pytorch uses the tensor and a.stride() operation to deal with that tensor

print('Stride of a? ',a.stride())

# Is aT contiguous a_transpose

print('Is aT contiguous? ',a.t().is_contiguous())
print('Stride of aT? ',a.t().stride())

# Lets check the data_ptr of a and aT
print('a data_ptr? ',a.data_ptr())
print('aT data_ptr? ',a.t().data_ptr())

# How view operation changes the tensor
# View operation requires the tensor to be contiguous
print('')
print('How view changes the requirement')
print('View requires the tensor to be contiguous let us use a.t() and reshape as it is non-contiguous')
print('What does reshape do, does it point to the same memory? ',a.t().reshape(15,1).data_ptr())
print('Memory for a only? ',a.t().data_ptr())
print('Is it contiguos after reshape? ',a.t().reshape(15,1).is_contiguous())
print('Will memory address change for all reshape? ',a.t().reshape(15,1).data_ptr())
print('Memory addres for a reshape, because it is contiguous does it still create new tensor? ',a.reshape(15,1).data_ptr())
print('It does not it does this only for non-contiguous tensors')


# To create a new copy of the tensor use a.clone() operation