#include "backend.hpp"

/**
 * @brief Get the name of the backend.
 * @param backend Backend.
 * @return Name of the backend.
 */
extern "C" const char* getBackendName(afft_Backend backend)
{
  return afft::toString(Convert<afft::Backend>::fromC(backend)).data();
}
