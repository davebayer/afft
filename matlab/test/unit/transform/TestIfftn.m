classdef TestIfftn < AbstractTestTransform
  properties (TestParameter)
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    precision     = {'single'; 'double'};
    complexity    = {'complex'}; % todo: add 'real' when implemented
    normalization = {'none'; 'unitary'; 'orthogonal'};
  end

  methods (Static)
    function dstRef = computeReference(src, normalization)
      dstRef = ifftn(src);

      if strcmp(normalization, 'none')
        dstRef = dstRef * numel(src);
      elseif strcmp(normalization, 'unitary')
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef * sqrt(numel(src));
      else
        error('Invalid normalization');
      end
    end
  end

  methods (Test)
    function testCpu(testCase, gridSize, precision, complexity, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, complexity, 'cpu');

      dstRef = TestIfftn.computeReference(src, normalization);
      dst    = afft.ifftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, gridSize, precision, complexity, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, complexity, 'gpu');

      dstRef = TestIfftn.computeReference(src, normalization);
      dst    = afft.ifftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end