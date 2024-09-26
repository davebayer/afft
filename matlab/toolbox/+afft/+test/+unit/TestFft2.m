classdef TestFft2 < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    precision     = {'single', 'double'};
    srcComplexity = {'complex'}; % todo: add 'real' when implemented
    normalization = {'none', 'unitary', 'orthogonal'};
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
  end

  methods (Static)
    function dstRef = computeReference(src, normalization)
      % Compute the reference using the built-in fft2 function.
      dstRef = fft2(src);

      transformSize = size(src, 1) * size(src, 2);

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
      elseif strcmp(normalization, 'unitary')
        dstRef = dstRef / transformSize;
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef / sqrt(transformSize);
      else
        error('Invalid normalization');
      end
    end
  end

  methods (Test)
    function testCpu(testCase, precision, srcComplexity, normalization, gridSize)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity);

      dstRef = TestFft2.computeReference(src, normalization);
      dst    = afft.fft2(src, 'normalization', normalization);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, srcComplexity, normalization, gridSize)
      src = gpuArray(AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity));

      dstRef = TestFft2.computeReference(src, normalization);
      dst    = afft.fft2(src, 'normalization', normalization);

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end