classdef TestDctn < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    dctType = {[], 1, 2, 3, 4};
  end

  methods (Static)
    function dstRef = computeReference(src, normalization, dctType)
      if strcmp(normalization, 'orthogonal')
        for dim = 1:ndims(src)
          if isempty(dctType)
            dstRef = dct(src, [], dim);
          else
            dstRef = dct(src, [], dim, Type=dctType);
          end
        end
      elseif strcmp(normalization, 'none') || strcmp(normalization, 'unitary')
        if isempty(dctType)
          dstRef = matlab.internal.math.transform.mldctn(gather(src));
        else
          dstRef = matlab.internal.math.transform.mldctn(gather(src), 'Variant', dctType);
        end

        if strcmp(normalization, 'unitary')
          dstRef = dstRef / numel(src);
        end
      else
        error('Invalid normalization');
      end
    end
  end

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, dctType)
      src = AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      dstRef = TestDctn.computeReference(src, normalization, dctType);

      if isempty(dctType)
        dst = afft.dctn(src, ...
                        'backend',       backend, ...
                        'normalization', normalization, ...
                        'threadLimit',   AbstractTestTransform.cpuThreadLimit);
      else
        dst = afft.dctn(src, ...
                        'type',          dctType, ...
                        'backend',       backend, ...
                        'normalization', normalization, ...
                        'threadLimit',   AbstractTestTransform.cpuThreadLimit);
      end

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, dctType)
      src = AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      try
        if isempty(dctType)
          dst = afft.dctn(src, ...
                          'backend',       backend, ...
                          'normalization', normalization, ...
                          'threadLimit',   AbstractTestTransform.cpuThreadLimit);
        else
          dst = afft.dctn(src, ...
                          'type',          dctType, ...
                          'backend',       backend, ...
                          'normalization', normalization, ...
                          'threadLimit',   AbstractTestTransform.cpuThreadLimit);
        end
        testCase.verifyFail('Expected afft.dctn to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, dctType)
      testCase.assumeTrue(afft.hasGpuSupport)

      testFailure(testCase, 'cufft', precision, normalization, gridSize, dctType);
    end

    function testFftw3(testCase, precision, normalization, gridSize, dctType)
      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, dctType);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, dctType);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, dctType)
      testFailure(testCase, 'mkl', precision, normalization, gridSize, dctType);
    end

    function testPocketfft(testCase, precision, normalization, gridSize, dctType)
      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, dctType);
    end

    function testVkfft(testCase, precision, normalization, gridSize, dctType)
      testCase.assumeTrue(afft.hasGpuSupport)
      
      if (strcmp(normalization, 'orthogonal') && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, dctType);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, dctType);
      end
    end
  end
end
