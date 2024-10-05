classdef TestIfft < afft.test.unit.AbstractTestTransform
  properties (Constant, TestParameter)
    symmetricFlag = {'nonsymmetric'}; % todo: implement symmetric
    dim           = {1}; % todo" add 1:afft.maxDimCount when implemented
  end

  methods (Static)
    function src = generateSrcArray(backend, gridSize, precision, symmetricFlag)
      % Generate a random array of the given size and precision.
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'complex');

      % Modify the array to match the given symmetric flag.
      if strcmp(symmetricFlag, 'symmetric')
        error('Symmetric flag not supported yet'); % todo: implement symmetric
      elseif strcmp(symmetricFlag, 'nonsymmetric')
      else
        error('Invalid symmetric flag');
      end
    end

    function dstRef = computeReference(src, dim, normalization, symmetricFlag)
      % Compute the reference using the built-in fft2 function.
      dstRef = ifft(src, [], dim, symmetricFlag);

      transformSize = size(src, dim);

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
        dstRef = dstRef * transformSize;
      elseif strcmp(normalization, 'unitary')
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef * sqrt(transformSize);
      else
        error('Invalid normalization');
      end
    end
  end

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, symmetricFlag, dim)
      src = afft.test.unit.TestIfft.generateSrcArray(backend, gridSize, precision, symmetricFlag);

      dstRef = afft.test.unit.TestIfft.computeReference(src, dim, normalization, symmetricFlag);
      dst    = afft.ifft(src, ...
                         [], ...
                         dim, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, symmetricFlag, dim)
      src = afft.test.unit.TestIfft.generateSrcArray(backend, gridSize, precision, symmetricFlag);

      try
        dst = afft.ifft(src, ...
                        [], ...
                        dim, ...
                        'backend',       backend, ...
                        'normalization', normalization, ...
                        'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        testCase.verifyFail('Expected afft.ifft to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, symmetricFlag, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('cufft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0) || (dim ~= 1 && dim ~= ndims(gridSize))
        testFailure(testCase, 'cufft', precision, normalization, gridSize, symmetricFlag, dim);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize, symmetricFlag, dim);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize, symmetricFlag, dim)
      testCase.assumeTrue(afft.hasBackend('fftw3'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, symmetricFlag, dim);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, symmetricFlag, dim);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, symmetricFlag, dim)
      testCase.assumeTrue(afft.hasBackend('mkl'));

      if dim ~= 1 && dim ~= ndims(gridSize)
        testFailure(testCase, 'mkl', precision, normalization, gridSize, symmetricFlag, dim);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize, symmetricFlag, dim);
      end
    end

    function testPocketfft(testCase, precision, normalization, gridSize, symmetricFlag, dim)
      testCase.assumeTrue(afft.hasBackend('pocketfft'));

      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, symmetricFlag, dim);
    end

    function testVkfft(testCase, precision, normalization, gridSize, symmetricFlag, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('vkfft'));

      if strcmp(normalization, 'orthogonal') && sum(gridSize) > 0
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, symmetricFlag, dim);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, symmetricFlag, dim);
      end
    end
  end
end
