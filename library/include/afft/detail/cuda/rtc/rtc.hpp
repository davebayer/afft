/*
  This file is part of afft library.

  Copyright (c) 2024 David Bayer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef AFFT_DETAIL_CUDA_RTC_RTC_HPP
#define AFFT_DETAIL_CUDA_RTC_RTC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../../include.hpp"
#endif

#include "error.hpp"
#include "../enviroment.hpp"
#include "../../common.hpp"
#include "../../utils.hpp"

namespace afft::detail::cuda::rtc
{
  /**
   * @struct CSymbolName
   * @brief A wrapper around a C symbol name.
   */
  struct CSymbolName : public std::string_view
  {
    /**
     * @brief Make all constructors explicit.
     * @tparam Args Argument types.
     * @param args Arguments to forward to the std::string_view constructor.
     */
    template<typename... Args>
    explicit constexpr CSymbolName(Args&&... args)
    : std::string_view(std::forward<Args>(args)...)
    {}

    // @brief Inherit all operators.
    using std::string_view::operator=;
    using std::string_view::operator[];
  };

  // Forward declaration.
  struct CppLoweredSymbolName;

  /**
   * @struct CppSymbolName
   * @brief A wrapper around a C++ symbol name.
   */
  struct CppSymbolName : public std::string_view
  {
    /**
     * @brief Make all constructors explicit.
     * @tparam Args Argument types.
     * @param args Arguments to forward to the std::string_view constructor.
     */
    template<typename... Args>
    explicit constexpr CppSymbolName(Args&&... args)
    : std::string_view(std::forward<Args>(args)...)
    {}

    /// @brief Disable construction from a CppLoweredSymbolName.
    CppSymbolName(const CppLoweredSymbolName&) = delete;

    /// @brief Inherit all operators.
    using std::string_view::operator=;
    using std::string_view::operator[];
  };

  /**
   * @class CppLoweredSymbolName
   * @brief A wrapper around a C++ lowered symbol name. This class can be created only by the Program class.
   */
  struct CppLoweredSymbolName : public std::string_view
  {
    /**
     * @brief Make all constructors explicit.
     * @tparam Args Argument types.
     * @param args Arguments to forward to the std::string_view constructor.
     */
    template<typename... Args>
    explicit constexpr CppLoweredSymbolName(Args&&... args)
    : std::string_view(std::forward<Args>(args)...)
    {}

    /// @brief Disable construction from a CppSymbolName.
    CppLoweredSymbolName(const CppSymbolName&) = delete;

    // @brief Inherit all operators.
    using std::string_view::operator=;
    using std::string_view::operator[];
  };

  /// @brief NVRTC code types.
  enum class CodeType
  {
    PTX,     ///< PTX code.
    CUBIN,   ///< CUBIN code.
    LTOIR,   ///< LTOIR code.
    OptixIR, ///< OptiX IR code.
  };

  /**
   * @class Code
   * @brief Program generated code.
   */
  class Code
  {
    public:
      /// @brief Default constructor.
      Code() = delete;

      /**
       * @brief Constructor.
       * @param type The code type.
       * @param size The code size.
       */
      Code(CodeType type, std::size_t size)
      : mType(type), mCode(size)
      {}

      /// @brief Copy constructor.
      Code(const Code&) = default;

      /// @brief Move constructor.
      Code(Code&&) = default;

      /// @brief Destructor.
      ~Code() = default;

      /// @brief Copy assignment operator.
      Code& operator=(const Code&) = default;

      /// @brief Move assignment operator.
      Code& operator=(Code&&) = default;

      /// @brief Get the code type.
      [[nodiscard]] CodeType type() const
      {
        return mType;
      }

      /// @brief Get the code data.
      [[nodiscard]] char* data()
      {
        return mCode.data();
      }

      /// @brief Get the code data.
      [[nodiscard]] const char* data() const
      {
        return mCode.data();
      }

      /// @brief Get the code size.
      [[nodiscard]] std::size_t size() const
      {
        return mCode.size();
      }
    protected:
    private:
      CodeType          mType{}; ///< The code type.
      std::vector<char> mCode{}; ///< The code.
  };

  /**
   * @class Program
   * @brief A program for the CUDA runtime compilation.
   */
  class Program
  {
    public:
      /// @brief Default constructor.
      Program() = delete;

      /**
       * @brief Constructor.
       * @tparam n The number of headers.
       * @param srcCode The source code of the program.
       * @param programName The name of the program.
       * @param headers The headers of the program.
       */
      Program(std::string_view srcCode, std::string_view programName)
      : mProgram{[&]()
      {
        nvrtcProgram program{};

        checkError(nvrtcCreateProgram(&program, srcCode.data(), programName.data(), 0, nullptr, nullptr));

        return program;
      }()}
      {}

      /// @brief Copy constructor.
      Program(const Program&) = delete;

      /// @brief Move constructor.
      Program(Program&&) = default;

      /// @brief Destructor.
      ~Program() = default;

      /// @brief Copy assignment operator.
      Program& operator=(const Program&) = delete;

      /// @brief Move assignment operator.
      Program& operator=(Program&&) = default;

      /**
       * @brief Add a name expression.
       * @param cppSymbolName The name of the symbol.
       */
      void addNameExpression(CppSymbolName cppSymbolName)
      {
        if (mIsCompiled)
        {
          throw std::runtime_error{"The program is already compiled"};
        }

        checkError(nvrtcAddNameExpression(mProgram.get(), cppSymbolName.data()));
      }

      /**
       * @brief Compile the program.
       * @param options The compilation options.
       * @return A tuple containing a boolean indicating if the compilation was successful and a string with the log.
       */
      [[nodiscard]] bool compile(Span<const char*> options)
      {
        if (std::exchange(mIsCompiled, true))
        {
          throw std::runtime_error{"The program is already compiled"};
        }

        bool ok = isOk(nvrtcCompileProgram(mProgram.get(), static_cast<int>(options.size()), options.data()));

        std::size_t logSize{};

        checkError(nvrtcGetProgramLogSize(mProgram.get(), &logSize));

        mCompilationLog.resize(logSize);

        checkError(nvrtcGetProgramLog(mProgram.get(), mCompilationLog.data()));

        return ok;
      }

      /**
       * @brief Get the lowered symbol name.
       * @param name The name of the symbol.
       * @return The lowered symbol name.
       */
      [[nodiscard]] CppLoweredSymbolName getLoweredSymbolName(CppSymbolName cppSymbolName)
      {
        if (!mIsCompiled)
        {
          throw std::runtime_error{"The program is not compiled"};
        }

        const char* loweredName{};

        checkError(nvrtcGetLoweredName(mProgram.get(), cppSymbolName.data(), &loweredName));

        return CppLoweredSymbolName{loweredName};
      }

      /**
       * @brief Get the code.
       * @param codeType The code type.
       * @return The code.
       */
      [[nodiscard]] Code getCode(CodeType codeType)
      {
        if (!mIsCompiled)
        {
          throw std::runtime_error{"The program is not compiled"};
        }

        std::size_t size{};
        switch (codeType)
        {
        case CodeType::PTX:     checkError(nvrtcGetPTXSize(mProgram.get(), &size));     break;
        case CodeType::CUBIN:   checkError(nvrtcGetCUBINSize(mProgram.get(), &size));   break;
        case CodeType::LTOIR:   checkError(nvrtcGetLTOIRSize(mProgram.get(), &size));   break;
        case CodeType::OptixIR: checkError(nvrtcGetOptiXIRSize(mProgram.get(), &size)); break;
        default:
          throw std::runtime_error{"Invalid code type"};
        }

        Code code{codeType, size};
        switch (codeType)
        {
        case CodeType::PTX:     checkError(nvrtcGetPTX(mProgram.get(), code.data()));     break;
        case CodeType::CUBIN:   checkError(nvrtcGetCUBIN(mProgram.get(), code.data()));   break;
        case CodeType::LTOIR:   checkError(nvrtcGetLTOIR(mProgram.get(), code.data()));   break;
        case CodeType::OptixIR: checkError(nvrtcGetOptiXIR(mProgram.get(), code.data())); break;
        default:
          throw std::runtime_error{"Invalid code type"};
        }

        return code;
      }

      [[nodiscard]] std::string_view getCompilationLog() const
      {
        return mCompilationLog;
      }
    protected:
    private:
      /**
       * @struct Deleter
       * @brief A deleter for the nvrtcProgram.
       */
      struct Deleter
      {
        /**
         * @brief Call operator.
         * @param program The program.
         */
        void operator()(nvrtcProgram program) const
        {
          nvrtcDestroyProgram(&program);
        }
      };

      std::unique_ptr<std::remove_pointer_t<nvrtcProgram>, Deleter> mProgram{};        ///< The program.
      bool                                                          mIsCompiled{};     ///< Program is compiled.
      std::string                                                   mCompilationLog{}; ///< The compilation log.
  };

  /**
   * @brief Get the supported architectures.
   * @return The supported architectures.
   */
  [[nodiscard]] inline std::vector<int> getSupportedArchs()
  {
    int size{};

    checkError(nvrtcGetNumSupportedArchs(&size));

    std::vector<int> archs(static_cast<std::size_t>(size));

    checkError(nvrtcGetSupportedArchs(archs.data()));

    return archs;
  }

  /**
   * @brief Make a real architecture option (e. g. sm_50).
   * @param device The CUDA device.
   * @return The real architecture option.
   */
  [[nodiscard]] inline std::string makeRealArchOption(int device)
  {
    const auto [ccMajor, ccMinor] = cuda::getComputeCapability(device);

    return cformat("-arch=sm_%d%d", ccMajor, ccMinor);
  }

  /**
   * @brief Make a relocatable device code option.
   * @param enable Enable the relocatable device code.
   * @return The relocatable device code option.
   */
  [[nodiscard]] inline std::string makeRelocatableDeviceCodeOption(bool enable)
  {
    return cformat("-rdc=%s", (enable ? "true" : "false"));
  }

  /**
   * @brief Make a debug option.
   * @return The debug option.
   */
  [[nodiscard]] inline std::string makeDebugOption()
  {
    return "-G";
  }

  /**
   * @brief Make a line info option.
   * @return The line info option.
   */
  [[nodiscard]] inline std::string makeLineInfoOption()
  {
    return "-lineinfo";
  }

  /**
   * @brief Make a fast math option.
   * @return The fast math option.
   */
  [[nodiscard]] inline std::string makeFastMathOption()
  {
    return "-use_fast_math";
  }
  
  /**
   * @brief Make a link time optimization option.
   * @return The link time optimization option.
   */
  [[nodiscard]] inline std::string makeLinkTimeOptimizationOption()
  {
    return "-dlto";
  }

  /**
   * @brief Make a preprocessor definition option.
   * @param name The name of the definition.
   * @param value The value of the definition.
   * @return The preprocessor definition option.
   */
  [[nodiscard]] inline std::string makeDefinitionOption(std::string_view name, std::string_view value)
  {
    return cformat("-D%s=%s", name.data(), value.data());
  }

  /**
   * @brief Make a preprocessor definition option.
   * @param name The name of the definition.
   * @param value The value of the definition.
   * @return The preprocessor definition option.
   */
  [[nodiscard]] inline std::string makeDefinitionOption(std::string_view name)
  {
    return cformat("-D%s", name.data());
  }

  /**
   * @brief Make an include path option.
   * @param includePath The include path.
   * @return The include path option.
   */
  [[nodiscard]] inline std::string makeIncludePathOption(std::string_view includePath)
  {
    return cformat("-I%s", includePath.data());
  }

  /**
   * @brief Make a C++ standard option.
   * @param version The C++ version.
   * @return The C++ standard option.
   */
  [[nodiscard]] inline std::string makeCppStandardOption(int version)
  {
    return cformat("-std=c++%d", version);
  }
} // namespace afft::detail::cuda::rtc

#endif /* AFFT_DETAIL_CUDA_RTC_RTC_HPP */
