/*
 *  IGeneric.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */


#include "IGeneric.h"

int IGeneric::tmpNextID = 0;

void IGeneric::setup(Interpolators *q, BasePixel *pix, int *path, int lp,  float startFrameDec) {
	isGeneric = true;
	id = tmpNextID++;
//	cout << "creating igeneric " << id << endl;
	needsReaping = false;
	timeScale = 1.0f;
      int startFrame = (int) startFrameDec;
    float dec = startFrameDec - startFrame;
    queue = q;

    pixel = pix;
    l = path;
    curI = (startFrame * 7)-7;
    t = -1;
    queue->add(this);
    loop = lp;

    if (dec > 0.0f) {
      advance();
      float dt = t * dec;
      curR += (dR * (float) dt);
      curG += (dG * (float) dt);
      curB += (dB * (float) dt);
      t -= dt;
    }

//    doneListener = NULL;
//    cou t << "IGeneric setup " << id << endl;
}


IGeneric::IGeneric(Interpolators *q, OffsetPixel *pix, int *path, int lp) : Interpolator() {
  if(pix != NULL)  {
    pix->setUser(this);
    setup(q, pix, path, lp, 0);
  } else {
    pixel = NULL;
    delete this;
  }


}

IGeneric::IGeneric(Interpolators *q, AmbientPixel *pix, int *path, int lp) : Interpolator() {
  if(pix != NULL)  {
    pix->setUser(this);
    setup(q, pix, path, lp, 0);
  } else {
    pixel = NULL;
    delete this;
  }

}

  IGeneric::IGeneric(Interpolators *q, BasePixel *pix, int *path, int lp) : Interpolator() {
  if(pix != NULL)  {
    setup(q, pix, path, lp, 0);
  } else {
    pixel = NULL;
    delete this;
  }
}

void IGeneric::notifyPixelDeletion(OffsetPixel *op) {
    pixel = NULL;
	needsReaping = true;
}

void IGeneric::notifyPixelDeletion(AmbientPixel *op) {
    pixel = NULL;
	needsReaping = true;
}

IGeneric::~IGeneric() {
	
//  cout << "IGeneric deleting " << id << endl;
  if (interpDoneListener != NULL) {
    interpDoneListener->genericInterpDone(this);
    interpDoneListener = NULL;
  }
  l = NULL;
  if (pixel != NULL) {
    if(pixel->pixelType == BasePixel::AMBIENT) {
      AmbientPixel *tmp = (AmbientPixel*)pixel;
      pixel = NULL;  // avoid circular reference
      tmp->setUser(NULL); // otherwise points back at self
      delete tmp;
	} else if (pixel->pixelType == BasePixel::OFFSET) {
		OffsetPixel *tmp = (OffsetPixel*) pixel;
		pixel = NULL;
		tmp->setUser(NULL);
		needsReaping = true;

		// don't delete offsetpixel will get reaped by avatar
//		delete tmp;
	}

  }
}

IGeneric::IGeneric(Interpolators *q, BasePixel *pix, int *path, int lp, float startFrameDec) {
  if(pix != NULL)  {
        setup(q, pix, path, lp, startFrameDec);
  } else {
    delete this;
  }
}

IGeneric::IGeneric(Interpolators *q, OffsetPixel *pix, int *path, int lp, float startFrameDec) {
//	cout << "igenric offset pixel creation" << endl;
 if(pix != NULL)  {
    pix->setUser(this);
        setup(q, pix, path, lp, startFrameDec);
  } else {
    delete this;
  }
}

IGeneric::IGeneric(Interpolators *q, AmbientPixel *pix, int *path, int lp, float startFrameDec) {
 if(pix != NULL)  {
    pix->setUser(this);
        setup(q, pix, path, lp, startFrameDec);
  } else {
    delete this;
  }
}

bool IGeneric::advance() {
  curI += 7;

  curR = l[curI + R1];
  if (curR <= -1) {
    if (loop <= 0) {
      curI = 0;
      curR = l[curI];
    } else if (loop == 1) {
      return false;
    } else {
      loop --;
      curI = 0;
      curR = l[curI];
    }
  }
  curG = l[curI + G1];
  curB = l[curI + B1];
  
  
  r2 = l[curI + R2];
  g2 = l[curI + G2];
  b2 = l[curI + B2];
  
  t = l[curI + T];

  
  if(t != 0) {
    float scale = 1.0f / (float) t;
    dR = ((float) (r2 - curR)) * scale;
    dG =((float)(g2 - curG)) * scale;
    dB =((float)(b2 - curB)) * scale;
  }
//  cou t << "advance " << curR << " " <<  curG << " " <<  curB << endl;
 // cou t << "        " <<  r2 << " " << g2 << " " <<b2 << "   "<< t << endl;

  
  return true;
  

}



//void IGeneric::addDoneListener(Avatar *a) {
//  doneListener = a;
//}
/*
void IGeneric::notifyDoneListener() {
  if (doneListener != NULL) {
    doneListener->genericInterpDone(this);
  }
}
*/


void IGeneric::update(int curTime, int deltaTime, float scale) {
  if (t <= 0) {
    if (! advance()) {
      if (pixel != NULL) {
		  if(pixel->pixelType == Pixel::AMBIENT) {
			  delete pixel;			  
		  } else
		  if(pixel->pixelType == Pixel::OFFSET) {
			  OffsetPixel *op = (OffsetPixel *) pixel;
			  op->needsReaping = true;
			  needsReaping = true;
			  return;
			 
//z			  delete pixel;
		  }
      }
	  needsReaping = true;
      return;
    }
  }
  if (pixel == NULL) {
	  needsReaping = true;
    return;
  } else {
    if(pixel->pixelType == Pixel::AMBIENT){ 
      pixel->setScale(scale);
    }
  }

  if (scale >= 0) { // have to check after advance (scale of 0 could destroy pixel if ambient or offset)
    if(curR > 0) {
      int b = true;
    }
    pixel->addColor( curR,  curG,  curB);      
    t -= deltaTime * timeScale;
    curR += (dR * (float) deltaTime);
    curG += (dG * (float) deltaTime);
    curB += (dB * (float) deltaTime);
  }
}
