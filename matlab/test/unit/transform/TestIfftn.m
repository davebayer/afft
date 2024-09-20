classdef TestIfftn < AbstractTestTransform
  properties (TestParameter)
    symmetricFlag = {'nonsymmetric'}; % todo: implement symmetric
  end

  methods (Static)
    function src = generateSrcArray(backend, gridSize, precision, symmetricFlag)
      % Generate a random array of the given size and precision.
      src = AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'complex');

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

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, symmetricFlag)
      src = AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'complex');

      dstRef = TestIfftn.computeReference(src, normalization, symmetricFlag);
      dst    = afft.ifftn(src, ...
                          symmetricFlag, ...
                          'backend',       backend, ...
                          'normalization', normalization, ...
                          'threadLimit',   AbstractTestTransform.cpuThreadLimit);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, symmetricFlag)
      src = AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'complex');

      try
        dst = afft.ifftn(src, ...
                         symmetricFlag, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   AbstractTestTransform.cpuThreadLimit);
        testCase.verifyFail('Expected afft.ifftn to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, symmetricFlag)
      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0) || numel(gridSize) > 3
        testFailure(testCase, 'cufft', precision, normalization, gridSize, symmetricFlag);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize, symmetricFlag);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize, symmetricFlag)
      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, symmetricFlag);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, symmetricFlag);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, symmetricFlag)
      if numel(gridSize) > 7
        testFailure(testCase, 'mkl', precision, normalization, gridSize, symmetricFlag);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize, symmetricFlag);
      end
    end

    function testPocketfft(testCase, precision, normalization, gridSize, symmetricFlag)
      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, symmetricFlag);
    end

    function testVkfft(testCase, precision, normalization, gridSize, symmetricFlag)
      if (strcmp(normalization, 'orthogonal') && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, symmetricFlag);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, symmetricFlag);
      end
    end
  end
end
