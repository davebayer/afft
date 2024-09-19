classdef TestDctn < AbstractTestTransform
  properties (TestParameter)
    precision     = {'single', 'double'};
    dctType       = {[], 1, 2, 3, 4};
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    normalization = {'none'}; % todo: add 'unitary', 'orthogonal' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, dctType, normalization)
      if isempty(dctType)
        dstRef = matlab.internal.math.transform.mldctn(gather(src));
      else
        dstRef = matlab.internal.math.transform.mldctn(gather(src), 'Variant', dctType);
      end

      % todo: implement orthogonal normalization

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
      elseif strcmp(normalization, 'unitary')
        dstRef = dstRef / numel(src);
      elseif strcmp(normalization, 'orthogonal')
        error('Orthogonal normalization not supported yet');
      else
        error('Invalid normalization');
      end
    end
  end

  methods (Test)
    function testCpu(testCase, precision, dctType, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real', 'cpu');

      dstRef = TestDctn.computeReference(src, dctType, normalization);
      
      if isempty(dctType)
        dst = afft.dctn(src); % todo: implement normalization
      else
        dst = afft.dctn(src, 'type', dctType); % todo: implement normalization
      end

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, dctType, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real', 'gpu');

      dstRef = TestDctn.computeReference(src, dctType, normalization);

      if isempty(dctType)
        dst = afft.dctn(src); % todo: implement normalization
      else
        dst = afft.dctn(src, 'type', dctType); % todo: implement normalization
      end

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end