#include "BasePixel.h"

BasePixel *BasePixel::blank = NULL;

BasePixel::BasePixel() { 
  x =  y = z = -10;
  isTarget = false;
  pixelType = BASE;
}