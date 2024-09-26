classdef TestDstn < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    precision     = {'single', 'double'};
    dstType       = {[], 1, 2, 3, 4};
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    normalization = {'none'}; % todo: add 'unitary', 'orthogonal' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, dstType, normalization)
      if isempty(dstType)
        dstRef = matlab.internal.math.transform.mldstn(gather(src));
      else
        dstRef = matlab.internal.math.transform.mldstn(gather(src), 'Variant', dstType);
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
    function testCpu(testCase, precision, dstType, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'real');

      dstRef = TestDstn.computeReference(src, dstType, normalization);
      
      if isempty(dstType)
        dst = afft.dstn(src); % todo: implement normalization
      else
        dst = afft.dstn(src, 'type', dstType); % todo: implement normalization
      end

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, dstType, gridSize, normalization)
      src = gpuArray(AbstractTestTransform.generateSrcArray(gridSize, precision, 'real'));

      dstRef = TestDstn.computeReference(src, dstType, normalization);

      if isempty(dstType)
        dst = afft.dstn(src); % todo: implement normalization
      else
        dst = afft.dstn(src, 'type', dstType); % todo: implement normalization
      end

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end