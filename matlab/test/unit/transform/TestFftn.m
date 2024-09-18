classdef TestFftn < AbstractTestTransform
  properties (TestParameter)
    precision     = {'single', 'double'};
    srcComplexity = {'complex'}; % todo: add 'real' when implemented
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    normalization = {'none'}; % todo: add 'unitary', 'orthogonal' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, normalization)
      % Compute the reference using the built-in fftn function.
      dstRef = fftn(src);

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
      elseif strcmp(normalization, 'unitary')
        dstRef = dstRef / numel(src);
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef / sqrt(numel(src));
      else
        error('Invalid normalization');
      end
    end
  end

  methods (Test)
    function testCpu(testCase, precision, srcComplexity, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity, 'cpu');

      dstRef = TestFftn.computeReference(src, normalization);
      dst    = afft.fftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, srcComplexity, gridSize, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity, 'gpu');

      dstRef = TestFftn.computeReference(src, normalization);
      dst    = afft.fftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end