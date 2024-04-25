#ifndef PNGWRITER_H_
#define PNGWRITER_H_

#include <stdint.h>

#if __cplusplus
  extern "C" {
#endif

  uint8_t *bytes_from_data(double *data, const int height, const int width,
                           const int num_channels, const char *fname,
                           const char lang);
  int save_png(double *data, const int nx, const int ny, const char *fname,
               const char lang);
  uint8_t *load_png(const char *fname, int *nx, int *ny, int *channels);
  void release_png(void *data);

#if __cplusplus
  }
#endif
#endif
