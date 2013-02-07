/*
 *  Panel.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */


#include "Panel.h"


Panel::Panel() {
  minX = FLT_MAX;
  minY = FLT_MAX;
  minZ = FLT_MAX;
  maxX = maxY = maxZ = -FLT_MAX;




}

Panel::Panel(int w, int h) {
  width = w;
  height = h;
  size = w * h;
  
  pixels = new BasePixel*[size];
//  memset(pixels, 0, (width * height) *  sizeof(BasePixel*)); // set to null
 for (int i = 0; i < (width * height); i++) {
    pixels[i] = new BasePixel();
  }

 minX = FLT_MAX;
 minY = FLT_MAX;
 minZ = FLT_MAX;
 maxX = maxY = maxZ = -FLT_MAX;
 
 distEast = new float[size];
 distSouth = new float[size];
 distSouthEast = new float[size];

 if ((w == 1) || (h == 1)) {
   isPillar = true;
 } else {
   isPillar = false;
 }

}


Panel::~Panel() {
  for(int i = 0; i < size; i++) {
    delete pixels[i];
  }
  delete[] pixels;
  delete[] distEast;
  delete[] distSouth;
  delete[] distSouthEast;
  delete[] xCalc;
  delete[] yCalc;

}
void Panel::calcACol() {
  int detailCols = (int) (maxX * SCALEFACTOR);    
  xCalc = new int[detailCols];
  
  int leftCol =  0;
  int rightCol = 1;
  
  
  for(int i = 0; i < detailCols; i++) {
    float leftX = -1.0f;
    int row = 0;
    while((leftX < 0) && (row < height)) {
      BasePixel *pixel = getPixel(leftCol, row);
      leftX = pixel->x;
      row++;
    }
    row = 0;
    float rightX = -1.0f;
    while((rightX < 0) && (row < height)) {
      BasePixel *pixel = getPixel(rightCol, row);
      rightX = pixel->x;
      row++;
    }
    
    float curX = ((float) i) * (1.0f / SCALEFACTOR);
    
    float dLeft = curX - leftX;
    float dRight = rightX - curX;
    
    
    ASSERT(((dLeft >= 0) || (leftCol == 0)), "dleft is less than 0 curX:" << curX << "  leftX:" << leftX << "  leftCol:" << leftCol);
    
    
    if (dLeft < dRight) {
      xCalc[i] = leftCol;
//      cou t << curX << " is closest to " << leftCol << endl;
    } else {
      xCalc[i] = rightCol;
 //     c out << curX << " is closest to " << rightCol << endl;
    }
    if(dRight <= 0) {
      
      leftCol++;
      rightCol++;
    }
  }
  
}
/*
void Panel::calcZ() {
  int detailCols = (int) (maxX * SCALEFACTOR);    
  zCalc = new int[detailCols];
  
  int botElevation = 0;
  int topElevation = 1;
  
  for(int i = 0; i < detailCols; i++) {
    float botH = getPixel(botElevation, 0)->z;
    float topH = getPixel(topElevation, 0)->z;
    float curH = detailCols * 0.1f;
    ASSERT(topH > botH, "calcZ is going backwards on sides");
    
    float dbot = curH - botH;
    float dtop = topH - curH;
    
    if (dbot < dtop) {
      zCalc[i] = botElevation;
      cou t << curH << " is closest to " << botElevation << endl;
    } else {
      zCalc[i] = topElevation;
      co ut << botH << " is closest to " << topElevation << endl;
    }
    if(dbot <= 0) {
      botElevation++;
      topElevation++;
    }
    
    
  }
  
  
  
}
*/
void Panel::calcARow() {
  int detailCols = (int) (maxY * SCALEFACTOR);    
  yCalc = new int[detailCols];
  
  int topRow =  0;
  int botRow = 1;
  
  
  for(int i = 0; i < detailCols; i++) {
    float topY = -1.0f;
    int col = 0;
    while((topY < 0) && (col < width)) {
      BasePixel *pixel = getPixel(col, topRow);
      topY = pixel->y;
      col++;
    }
    col = 0;
    float botY = -1.0f;
    while((botY < 0) && (col < width)) {
      BasePixel *pixel = getPixel(col, botRow);
      botY = pixel->y;
      col++;
    }
    
    float curY = ((float) i) * (1.0f / SCALEFACTOR);
    
    float dTop = curY - topY;
    float dBot = botY - curY;
    
    
    ASSERT(((dTop >= 0) || (topRow == 0)), "dTop is less than 0 curY:" << curY << "  topY:" << topY << "  botRow:" << topRow);
    
    
    if (dTop < dBot) {
      yCalc[i] = topRow;
//      co ut << curY << " is closest to " << topRow << endl;
    } else {
      yCalc[i] = botRow;
//      co ut << curY << " is closest to " << botRow << endl;
    }
    if(dBot <= 0) {
      topRow++;
      botRow++;
    }
  }
  
}



void Panel::calcStats() {
  if(minZ == maxZ) { // if A
    calcARow();
    calcACol();
    calcDistGrid();
  } else if (width > 1) { // b
    calcARow();
    //calcZ();
  } else {
    //calcZ();
  }
  
}

void Panel::calcDistGrid() {
  // calc East;
	int r;
  for(r = 0; r < height; r++) {
    int curRow = r*width;
    for(int c = 0; c < width-1; c++) {
      distEast[c + curRow] = pixels[(c+1) + curRow]->x - pixels[c + curRow]->x;
    }
    distEast[width-1 + curRow] = INT_MAX; // something really big so you don't interpolate past pixel
  }

   // calc South;
  for(r = 0; r < height - 1; r++) {
    int curRow = r*width;
    int nextRow = (r*width) + 1;
    for(int c = 0; c < width; c++) {
      distEast[c + curRow] = pixels[c+nextRow]->y - pixels[c + curRow]->x;
    }
  }
    for(int c = 0; c < width; c++) {
      distEast[c + height - 1] = FLT_MAX;
    }

}

 
int Panel::getCol(float x) {
  if (minX == maxX) {
    return -1;
  } else {
    if (x < 0) {
      return 0;
    } else if (x > maxX) {
      return (width -1);
    } else {
      return xCalc[(int) (x * SCALEFACTOR)];    
    }
  }
}

int Panel::getRow(float y) {
  if (width > 1) {
    if (y < 0) {
      return 0;
    } else if (y < maxY) {
      return yCalc[(int) (y * SCALEFACTOR)];
    } else {
      return height -1;
    }
  } else {
    return -1;
  }
}
/*
int Panel::getElevation(float z) {
  if(minZ == maxZ) {
    return -1;
  } else {
    if (z < 0) {
      return 0;
    } else if (z > maxZ) {
      return height -1;
    } else {
      return yCalc[(int) (z * SCALEFACTOR)];
    }
  }
  
}
*/
void Panel::set(int w, int h) {
  width = w;
  height = h;
  size = w * h;
  
  pixels = new BasePixel*[width * height];
  //  memset(pixels, 0, (width * height) *  sizeof(BasePixel*)); // set to null
  for (int i = 0; i < (width * height); i++) {
    pixels[i] = new BasePixel();
  }

  distEast = new float[size];
 distSouth = new float[size];
 distSouthEast = new float[size];

  if ((w == 1) || (h == 1)) {
   isPillar = true;
 } else {
   isPillar = false;
 }


}
void Panel::display() {
  for(int i = 0; i < size; i++) {
    pixels[i]->display();
  }

}
void Panel::topDisplay() {
  for(int i = 0; i < size; i++) {
    pixels[i]->topDisplay();
  }
  
}

void Panel::update() {
  for(int i = 0; i < size; i++) {
    pixels[i]->update();
  }
 
  
}


void Panel::setPixel(int col, int row, BasePixel *pixel) {
  int i;
  if(isPillar) {
    ASSERT((col == 0) || (row == 0), "Panel.setPixel either row or col must be 0 for pillars");
    ASSERT((col + row) < size, "Panel.setPixel either row or col must be 0 for pillars");
     i = col + row;
   

  } else {
    ASSERT(col < width, "Panel.setPixel: column " << col << " is greather than width " << width);
    ASSERT(col >= 0, "Panel.setPixel: column out of bounds");
    ASSERT(row < height, "Panel.setPixel: row out of bounds");
    ASSERT(row >= 0, "Panel.setPixel: row is negative");
     i = col + (row * width);
  }

    delete pixels[i];
     pixels[i] = pixel;
   
     minX = (minX < pixel->x) ? minX : pixel->x; 
     minY = (minY < pixel->y) ? minY : pixel->y; 
     minZ = (minZ < pixel->z) ? minZ : pixel->z; 
   
     maxX = (maxX > pixel->x) ? maxX : pixel->x; 
     maxY = (maxY > pixel->y) ? maxY : pixel->y; 
     maxZ = (maxZ > pixel->z) ? maxZ : pixel->z; 

   
}



BasePixel* Panel::getPixel(int col, int row) {
  if(isPillar) { // col or row should be 0
    if ((col != 0) && (row != 0)) return BasePixel::blank;
    int i = row + col;
    if (i >= size) return BasePixel::blank;
    return pixels[i];
  } else {
    if (col >= width) return BasePixel::blank;
    if (col < 0) return BasePixel::blank;
    if (row >= height) return BasePixel::blank;
    if (row < 0) return BasePixel::blank;

    return pixels[col + (row * width)];
  }
}

float Panel::floorDistSquared(float x, float y, float row) {
  BasePixel* pix = getPixel(0, row);
  float dx = x - pix->x;
  float dy = y - pix->y;
  return dx * dx + dy * dy;


}
BasePixel* Panel::getPixel(int row) {
  if ((width != 1) || (height != 1)) {  // getting a row is only valid on columns
    return BasePixel::blank;
  } else if (row >= ((width + height) - 1)) {
    return BasePixel::blank;
  } else if (row < 0) {
    return BasePixel::blank;
  } else {
  return pixels[row];  
  }
}

BasePixel* Panel::getPixelRow(int row) {
  if (row >= height) {
    return BasePixel::blank;
  } else if (row < 0) {
    return BasePixel::blank;
  } else {
    return pixels[row];  
  }
}