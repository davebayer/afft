function res = isNormalization(str)
  res = ischar(str) && ismember(str, {'none', 'unitary', 'orthogonal'});
end
