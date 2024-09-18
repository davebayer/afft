classdef TestIfftn < AbstractTestTransform
  properties (TestParameter)
    precision     = ['single'; 'double'];
    gridSize      = [AbstractTestTransform.GridSizes0D, ...
                     AbstractTestTransform.GridSizes1D, ...
                     AbstractTestTransform.GridSizes2D, ...
                     AbstractTestTransform.GridSizes3D];
    symmetricFlag = ['nonsymmetric']; % todo: implement symmetric
    normalization = ['none'; 'unitary'; 'orthogonal'];
  end

  methods (Static)
    function src = generateSrcArray(gridSize, precision, complexity, target, symmetricFlag)
      % Generate a random array of the given size and precision.
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, complexity, target);

      % Modify the array to match the given symmetric flag.
      if strcmp(symmetricFlag, 'symmetric')
        error('Symmetric flag not supported yet'); % todo: implement symmetric
      elseif strcmp(symmetricFlag, 'nonsymmetric')
      else
        error('Invalid symmetric flag');
      end
    end

    function dstRef = computeReference(src, normalization, symmetricFlag)
      % Compute the reference using the built-in ifftn function.
      dstRef = ifftn(src, symmetricFlag);

      % Modify the reference to match the given normalization.
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
    function testCpu(testCase, precision, gridSize, symmetricFlag, normalization)
      src = TestIfftn.generateSrcArray(gridSize, precision, 'complex', 'cpu', symmetricFlag);

      dstRef = TestIfftn.computeReference(src, normalization);
      dst    = afft.ifftn(src); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, gridSize, symmetricFlag, normalization)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, 'complex', 'gpu', symmetricFlag);

      dstRef = TestIfftn.computeReference(src, normalization);
      dst    = afft.ifftn(src, symmetricFlag); % todo: implement normalization

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end