function res = isDimension(dim)
  res = isnumeric(dim) && isscalar(dim) && isreal(dim) && dim > 0 && dim == round(dim);
end
